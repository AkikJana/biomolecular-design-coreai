"""A local screening tool for the bench: paste a target and candidates, get a ranking.

This is the project's findings turned into something usable rather than a summary
of them. Three of those findings are built into the tool rather than written in a
caveat somewhere:

**The scramble control is not optional.** Section 7.4 showed a confidence score
can separate a peptide from an unrelated decoy while being blind to whether the
sequence is in the right order -- it was reading composition. So for every
candidate this tool also folds permutations *of that same candidate*, and reports
the candidate against its own null. A candidate that does not beat its own
scrambles is not a hit, whatever its raw score.

**Reduced settings are reported as such.** Section 7.13 measured the cost: at 10
sampling steps the effect is three to seven times smaller in standardised terms
and only 14% of backbone bonds are physically plausible. Quick mode is offered
because it is 3-4x faster, and it says plainly what it costs.

**A single fold is not a measurement.** Section 7.5 found per-receptor rankings
flip between identical re-runs. Replicates are a first-class control here, and
the interface reports the spread rather than one number.

No new dependencies: stdlib http.server, so it runs wherever the project runs.

Usage:
    python src/screen_server.py            # then open http://127.0.0.1:8765
"""

import argparse
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

ART = REPO_ROOT / "artifacts"
WORK = ART / "screen_jobs"
UI = REPO_ROOT / "demo" / "app.html"
AA = set("ACDEFGHIKLMNPQRSTVWY")

MODES = {
    "quick": {"label": "Quick screen", "model": "decaf", "sampling": 10,
              "recycling": 1, "msa_depth": 32, "sec_per_fold": 30},
    "careful": {"label": "Careful screen", "model": "boltz1", "sampling": 200,
                "recycling": 3, "msa_depth": None, "sec_per_fold": 110},
}

JOBS = {}
JOBS_LOCK = threading.Lock()
WORKQ = queue.Queue()


# ----------------------------------------------------------------- validation

def check_peptide(seq):
    """Reasons this candidate cannot be scored honestly, or []."""
    problems = []
    bad = sorted(set(seq) - AA)
    if bad:
        problems.append(f"non-standard residues {','.join(bad)} — RCSB writes X for "
                        f"anything that is not one of the twenty, and the model cannot "
                        f"fold what it cannot name")
    if len(seq) < 4:
        problems.append("shorter than 4 residues")
    if len(seq) > 40:
        problems.append("longer than 40 residues — outside the 6–25 range this was "
                        "benchmarked on")
    return problems


def scrambles_of(seq, n, rng):
    """Permutations of the candidate: same composition and length, order destroyed."""
    out, seen, tries = [], {seq}, 0
    while len(out) < n and tries < 200:
        tries += 1
        s = list(seq)
        rng.shuffle(s)
        s = "".join(s)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ----------------------------------------------------------------- folding

def fetch_msa(receptor, job_dir):
    """Alignment for the target, cached by sequence so a re-screen is instant."""
    import hashlib
    key = hashlib.sha1(receptor.encode()).hexdigest()[:16]
    cache = ART / "screen_msa_cache"
    cache.mkdir(parents=True, exist_ok=True)
    hit = cache / f"{key}.csv"
    if hit.exists():
        return str(hit), True
    probe = job_dir / "msa" / "inputs"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "probe.yaml").write_text(
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {receptor}\n"
        f"  - protein:\n      id: B\n      sequence: AAAAAA\n      msa: empty\n")
    cmd = [sys.executable, "-m", "boltz.main", "predict", str(probe),
           "--out_dir", str(probe.parent), "--model", "boltz1",
           "--accelerator", "cpu", "--recycling_steps", "0", "--sampling_steps", "5",
           "--output_format", "pdb", "--override", "--use_msa_server",
           "--subsample_msa", "--num_subsampled_msa", "32", "--max_msa_seqs", "32",
           "--num_workers", "0"]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    src = probe.parent / "boltz_results_inputs" / "msa" / "probe_0.csv"
    if src.exists():
        shutil.copy2(src, hit)
        return str(hit), False
    return None, False


def fold_batch(pairs, job_dir, mode, msa_path, budget_per_fold):
    """Fold a list of (name, receptor, peptide). Returns {name: readouts}."""
    from Bio.PDB import PDBParser
    from interface_side_split import sides

    cfg = MODES[mode]
    inputs = job_dir / "inputs"
    shutil.rmtree(inputs, ignore_errors=True)
    inputs.mkdir(parents=True)
    for name, rec, pep in pairs:
        line = f"      msa: {msa_path}\n" if msa_path else "      msa: empty\n"
        (inputs / f"{name}.yaml").write_text(
            "version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {rec}\n{line}"
            f"  - protein:\n      id: B\n      sequence: {pep}\n      msa: empty\n")

    if cfg["model"] == "decaf":
        cmd = [sys.executable, str(REPO_ROOT / "src" / "decaf_runner.py"), "predict",
               str(inputs), "--checkpoint",
               str(Path.home() / ".boltz" / "decaf" / "decaf_conf_ckpt.ckpt"),
               "--model", "boltz1"]
    else:
        cmd = [sys.executable, "-m", "boltz.main", "predict", str(inputs),
               "--model", "boltz1"]
    cmd += ["--out_dir", str(job_dir), "--accelerator", "gpu",
            "--output_format", "pdb", "--override", "--diffusion_samples", "1",
            "--recycling_steps", str(cfg["recycling"]),
            "--sampling_steps", str(cfg["sampling"]),
            "--num_workers", "0", "--preprocessing-threads", "1"]
    if cfg["msa_depth"] is not None:
        cmd += ["--subsample_msa", "--num_subsampled_msa", str(cfg["msa_depth"]),
                "--max_msa_seqs", str(cfg["msa_depth"])]
    if cfg["model"] == "decaf":
        cmd.append("--no_kernels")

    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=max(900, len(pairs) * budget_per_fold))
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout + proc.stderr)[-1500:])

    parser = PDBParser(QUIET=True)
    res = job_dir / f"boltz_results_{inputs.name}" / "predictions"
    out = {}
    for name, _, _ in pairs:
        pdb = res / name / f"{name}_model_0.pdb"
        conf = res / name / f"confidence_{name}_model_0.json"
        if not pdb.exists():
            continue
        try:
            s = sides(parser.get_structure("x", str(pdb))[0])
        except Exception:
            continue
        if s:
            s["iptm"] = (json.loads(conf.read_text()).get("iptm")
                         if conf.exists() else None)
            out[name] = s
    return out


# ----------------------------------------------------------------- scoring

def pooled_null_sd(job):
    """One spread estimate pooled over every candidate's scrambles.

    A per-candidate SD from two or three permutations is worthless as a
    denominator: when two scrambles happen to land close together it collapses
    and the z explodes. A first run of this tool reported z = +77 and -80 on a
    two-scramble job, which is not a number anyone should act on.

    Pooling the within-candidate deviations gives an estimate with real degrees
    of freedom -- three candidates at two scrambles each yields df = 3 rather
    than 1 -- and is the standard treatment when each group is tiny.
    """
    import numpy as np
    dev, groups = [], 0
    for cand in job["candidates"]:
        vals = [x for r in cand["null_reps"] for x in r if x is not None]
        if len(vals) >= 2:
            m = float(np.mean(vals))
            dev += [v - m for v in vals]
            groups += 1
    df = len(dev) - groups
    if df < 2:
        return None, df
    return float(np.sqrt(sum(d * d for d in dev) / df)), df


def summarise(job):
    """Rank candidates against their own scrambles, with replicate spread."""
    import numpy as np
    metric = job["metric"]
    pooled, pooled_df = pooled_null_sd(job)
    job["pooled_null_sd"] = pooled
    job["pooled_df"] = pooled_df
    rows = []
    for cand in job["candidates"]:
        if cand.get("problems"):
            rows.append({**cand, "status": "not scored"})
            continue
        got = [r for r in cand["reps"] if r is not None]
        nulls = [x for r in cand["null_reps"] for x in r if x is not None]
        if not got or not nulls:
            rows.append({**cand, "status": "failed"})
            continue
        v = float(np.mean(got))
        spread = float(np.std(got, ddof=1)) if len(got) > 1 else None
        nm = float(np.mean(nulls))
        # A single null value has no spread to estimate, and np.std(ddof=1)
        # returns NaN there -- which `or 1e-9` does not catch, because NaN is
        # truthy. Rather than invent a denominator, the z is withheld and the
        # raw margin reported instead.
        ns = float(np.std(nulls, ddof=1)) if len(nulls) > 1 else None
        # the pooled spread is the denominator; a per-candidate SD from two or
        # three permutations is too unstable to divide by
        z = ((v - nm) / pooled) if (pooled and pooled > 1e-6) else None
        if z is None:
            verdict = (f"not enough scrambles to estimate a null "
                       f"(margin {v - nm:+.2f})")
        elif z >= 1.0:
            verdict = "beats its own scrambles"
        else:
            verdict = "indistinguishable from its own scrambles"
        rows.append({**cand, "status": "scored", "score": v,
                     "replicate_sd": spread, "null_mean": nm, "null_sd": ns,
                     "margin": v - nm, "z_vs_own_scrambles": z,
                     "pooled_null_sd": pooled,
                     "n_reps": len(got), "n_null": len(nulls),
                     "verdict": verdict})
    scored = [r for r in rows if r["status"] == "scored"]
    # rank on z where it exists, else on the raw margin
    scored.sort(key=lambda r: -(r["z_vs_own_scrambles"]
                                if r["z_vs_own_scrambles"] is not None
                                else r["margin"]))
    for i, r in enumerate(scored, 1):
        r["rank"] = i
    return scored + [r for r in rows if r["status"] != "scored"], metric


def run_job(job_id):
    with JOBS_LOCK:
        job = JOBS[job_id]
    job_dir = WORK / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    cfg = MODES[job["mode"]]

    def step(msg, done=None):
        with JOBS_LOCK:
            job["message"] = msg
            if done is not None:
                job["folds_done"] = done

    try:
        step("fetching alignment for the target …")
        msa, cached = fetch_msa(job["receptor"], job_dir)
        step("alignment ready (cached)" if cached else "alignment ready")
        with JOBS_LOCK:
            job["msa"] = "cached" if cached else ("fetched" if msa else "none")

        # every candidate gets its own permutations; that is the whole point
        pairs, index = [], {}
        for ci, cand in enumerate(job["candidates"]):
            if cand.get("problems"):
                continue
            for rep in range(job["replicates"]):
                n = f"c{ci}_r{rep}"
                pairs.append((n, job["receptor"], cand["peptide"]))
                index[n] = (ci, "reps", rep)
            for si, sc in enumerate(cand["scrambles"]):
                for rep in range(job["replicates"]):
                    n = f"c{ci}_s{si}_r{rep}"
                    pairs.append((n, job["receptor"], sc))
                    index[n] = (ci, "null_reps", si)

        with JOBS_LOCK:
            job["folds_total"] = len(pairs)
        done = 0
        batch = 6
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            step(f"folding {done + 1}–{done + len(chunk)} of {len(pairs)} "
                 f"({cfg['label'].lower()}) …", done)
            got = fold_batch(chunk, job_dir, job["mode"], msa,
                             cfg["sec_per_fold"] * 4)
            for name, _, _ in chunk:
                if name not in got:
                    continue
                ci, kind, slot = index[name]
                cand = job["candidates"][ci]
                val = got[name].get(job["metric"])
                if kind == "reps":
                    cand["reps"].append(val)
                else:
                    cand["null_reps"][slot].append(val)
            done += len(chunk)
            step(f"folded {done} of {len(pairs)}", done)
            shutil.rmtree(job_dir / "boltz_results_inputs", ignore_errors=True)

        results, metric = summarise(job)
        with JOBS_LOCK:
            job["results"] = results
            job["state"] = "done"
            job["message"] = f"complete — {len(pairs)} folds"
            job["finished_at"] = time.time()
    except Exception as exc:                                   # noqa: BLE001
        with JOBS_LOCK:
            job["state"] = "failed"
            job["message"] = str(exc)[:400]
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def worker():
    while True:
        job_id = WORKQ.get()
        if job_id is None:
            return
        run_job(job_id)
        WORKQ.task_done()


# ----------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                                 # quiet console
        pass

    @staticmethod
    def _json(obj):
        """json.dumps emits bare NaN/Infinity, which JSON.parse rejects."""
        import math

        def clean(o):
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return o
        return json.dumps(clean(obj), allow_nan=False)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if not UI.exists():
                return self._send(500, "app.html missing — run build_demo_site.py",
                                  "text/plain")
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/config":
            return self._send(200, json.dumps({
                "modes": {k: {"label": v["label"], "sampling": v["sampling"],
                              "recycling": v["recycling"],
                              "msa": "full" if v["msa_depth"] is None else v["msa_depth"],
                              "sec_per_fold": v["sec_per_fold"]}
                          for k, v in MODES.items()}}))
        if self.path.startswith("/api/job/"):
            jid = self.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(jid)
                if not job:
                    return self._send(404, json.dumps({"error": "no such job"}))
                view = {k: job[k] for k in
                        ("state", "message", "folds_done", "folds_total", "mode",
                         "replicates", "metric", "msa", "pooled_null_sd",
                         "pooled_df") if k in job}
                view["results"] = job.get("results")
                view["elapsed"] = round(time.time() - job["started_at"], 1)
            return self._send(200, self._json(view))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/screen":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, json.dumps({"error": "bad JSON"}))

        receptor = "".join((req.get("receptor") or "").split()).upper()
        peptides = [("".join(p.split())).upper()
                    for p in (req.get("peptides") or []) if p.strip()]
        mode = req.get("mode", "quick")
        reps = max(1, min(5, int(req.get("replicates", 2))))
        nulls = max(1, min(4, int(req.get("scrambles", 2))))
        metric = req.get("metric", "iface_plddt")

        if not receptor or len(set(receptor) - AA) or len(receptor) < 30:
            return self._send(400, json.dumps({
                "error": "target must be 30+ residues of the standard twenty"}))
        if not peptides:
            return self._send(400, json.dumps({"error": "no candidates given"}))
        if len(peptides) > 12:
            return self._send(400, json.dumps({
                "error": "12 candidates max per job on this hardware"}))
        if mode not in MODES:
            return self._send(400, json.dumps({"error": "unknown mode"}))

        rng = random.Random(0)
        cands = []
        for p in peptides:
            probs = check_peptide(p)
            cands.append({"peptide": p, "problems": probs,
                          "scrambles": [] if probs else scrambles_of(p, nulls, rng),
                          "reps": [], "null_reps": [[] for _ in range(nulls)]})

        jid = uuid.uuid4().hex[:12]
        n_folds = sum((1 + nulls) * reps for c in cands if not c["problems"])
        with JOBS_LOCK:
            JOBS[jid] = {"id": jid, "state": "queued", "message": "queued",
                         "receptor": receptor, "candidates": cands, "mode": mode,
                         "replicates": reps, "metric": metric,
                         "folds_total": n_folds, "folds_done": 0,
                         "started_at": time.time()}
        WORKQ.put(jid)
        return self._send(200, json.dumps({
            "job_id": jid, "folds": n_folds,
            "eta_seconds": int(n_folds * MODES[mode]["sec_per_fold"] + 120)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    srv = HTTPServer((args.host, args.port), Handler)
    print(f"Peptide screening tool  →  http://{args.host}:{args.port}")
    print(f"  quick   : DeCAF, 10 steps   ~{MODES['quick']['sec_per_fold']}s/fold")
    print(f"  careful : Boltz-1, 200 steps ~{MODES['careful']['sec_per_fold']}s/fold")
    print("  every candidate is folded against permutations of itself")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
