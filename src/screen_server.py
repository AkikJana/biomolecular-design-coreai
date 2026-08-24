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
MSA_CACHE = ART / "screen_msa_cache"
FOLD_CACHE = ART / "screen_fold_cache"
UI = REPO_ROOT / "demo" / "app.html"
AA = set("ACDEFGHIKLMNPQRSTVWY")

# sec_per_fold is the MARGINAL cost of one more fold, not the cost of a fold in
# isolation. Model construction is ~45s and is paid once for the whole job, so a
# 15-fold quick screen is 45 + 15*9 rather than 15*54.
FIXED_STARTUP_SEC = 45
MODES = {
    "quick": {"label": "Quick screen", "model": "decaf", "sampling": 10,
              "recycling": 1, "msa_depth": 32, "sec_per_fold": 9},
    "careful": {"label": "Careful screen", "model": "boltz1", "sampling": 200,
                "recycling": 3, "msa_depth": None, "sec_per_fold": 33},
}

JOBS = {}
JOBS_LOCK = threading.Lock()
WORKQ = queue.Queue()

# Receptors and cognate peptides are the panel's own, so a preset runs on the
# sequences the reported results were measured on. Kept here rather than in the
# page so the buttons and demo/EXAMPLES.md cannot drift apart.
_MDM2 = ("SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCS"
         "NDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV")
_SH3 = "AEYVRALFDFNGNDEEDLPFKKGDILRIRDKPEEQWWNAEDSEGKRGMIPVPYVEKY"
_PDZ = ("GSPEFLGEEDIPREPRRIVIHRGSTGLGFNIIGGEDGEGIFISFILAGGPADLSGELRKGDQ"
        "ILSVNGVDLRNASHEQAAIALKNAGQTVTIIAQYKPEEYSRFEANSRVNSSGRIVTN")
_THREE = ["SQETFSDLWKLLPEN", "PPPALPPKKR", "KQTSV"]

EXAMPLES = [
    {"id": "mdm2", "name": "MDM2 · p53", "target": "MDM2 (1YCR)",
     "note": "p53's helix first at p = 0.002. PEPTIDEX is refused before folding.",
     "receptor": _MDM2, "peptides": _THREE + ["TSFAEYWNLLSP", "PEPTIDEX"],
     "mode": "quick", "replicates": 2, "scrambles": 3},
    {"id": "pdz", "name": "PDZ · same candidates", "target": "PSD-95 PDZ3 (1BE9)",
     "note": "The same three candidates, opposite answer: KQTSV first at "
             "p < 0.001, and the helix that won on MDM2 at p = 0.48.",
     "receptor": _PDZ, "peptides": _THREE,
     "mode": "quick", "replicates": 2, "scrambles": 3},
    {"id": "mdm2_careful", "name": "MDM2 · full sampling", "target": "MDM2 (1YCR)",
     "note": "What 200 steps buys: the null tightens 5.36 → 2.65 and both real "
             "binders reach p < 0.001.",
     "receptor": _MDM2, "peptides": _THREE + ["TSFAEYWNLLSP"],
     "mode": "careful", "replicates": 2, "scrambles": 5},
    {"id": "sh3", "name": "SH3 · a negative", "target": "c-Crk SH3 (1CKA)",
     "note": "The cognate does not separate from its own scrambles (p = 0.40), "
             "and full sampling does not rescue it. Kept because it is honest.",
     "receptor": _SH3, "peptides": _THREE,
     "mode": "quick", "replicates": 2, "scrambles": 3},
]


def example_readiness(ex):
    """Folds this preset still needs, so a button can say whether it is instant."""
    cfg = MODES[ex["mode"]]
    need = have = 0
    for pep in ex["peptides"]:
        if check_peptide(pep):
            continue
        for seq, n in ([(pep, ex["replicates"])]
                       + [(s, 1) for s in scrambles_of(pep, ex["scrambles"])]):
            need += n
            have += min(n, len(cache_read(fold_key(ex["receptor"], seq, cfg))))
    return need, have


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


def scrambles_of(seq, n, rng=None):
    """Permutations of the candidate: same composition and length, order destroyed.

    The stream is seeded from the peptide itself, so a candidate's null depends
    only on that candidate. Drawing every scramble from one job-level generator
    meant the permutations a peptide got depended on which other peptides were
    in the box and in what order -- so adding a fourth candidate silently
    changed the first three's nulls, and their scores moved between runs for no
    reason the user could see. Being peptide-local also makes the null
    reproducible across jobs, and lets the fold cache hit.
    """
    import hashlib
    if rng is None:
        rng = random.Random(hashlib.sha1(seq.encode()).hexdigest())
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
    """Alignment for the target, cached by sequence so a re-screen is instant.

    This calls the alignment server directly rather than running a throwaway
    fold to get the file as a side effect. The old route started a whole boltz
    process on CPU -- ~47s of model construction -- purely to reach the MSA step
    and then discard the structure it went on to predict.
    """
    import hashlib
    key = hashlib.sha1(receptor.encode()).hexdigest()[:16]
    MSA_CACHE.mkdir(parents=True, exist_ok=True)
    hit = MSA_CACHE / f"{key}.csv"
    if hit.exists():
        return str(hit), True
    try:
        sys.path.insert(0, str(Path.home() / ".boltz" / "decaf" / "repo" / "src"))
        from boltz.data.msa.mmseqs2 import run_mmseqs2
        tmp = job_dir / "msa"
        tmp.mkdir(parents=True, exist_ok=True)
        a3m = run_mmseqs2([receptor], str(tmp), use_env=True, use_filter=True)[0]
        rows = ["key,sequence"]
        for line in a3m.splitlines():
            if line.startswith(">") or not line.strip():
                continue
            # a3m marks insertions relative to the query in lower case; dropping
            # them leaves every row the query's width, which is what boltz reads
            aligned = "".join(c for c in line.strip() if not c.islower())
            if aligned:
                rows.append(f"-1,{aligned}")
        if len(rows) > 1:
            hit.write_text("\n".join(rows) + "\n")
            return str(hit), False
    except Exception:                                              # noqa: BLE001
        pass
    return None, False


# ----------------------------------------------------------------- fold cache

def fold_key(receptor, peptide, cfg):
    """Identity of a fold: same inputs *and* same settings, or it is not the same."""
    import hashlib
    stamp = f"{receptor}|{peptide}|{cfg['model']}|{cfg['sampling']}|" \
            f"{cfg['recycling']}|{cfg['msa_depth']}"
    return hashlib.sha1(stamp.encode()).hexdigest()[:20]


def cache_read(key):
    """Every independent fold ever run for this key, as a list.

    Storing a list rather than one value is the whole point. Folds are unseeded,
    so replicates are meant to differ; collapsing a key to a single cached value
    would hand every replicate the same number, drive the replicate spread to
    exactly zero, and make the tool claim a reproducibility it has not got.
    Cached entries are genuine independent folds, so a replicate may draw one.
    """
    f = FOLD_CACHE / f"{key}.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except Exception:                                              # noqa: BLE001
        return []


def cache_append(key, readouts):
    FOLD_CACHE.mkdir(parents=True, exist_ok=True)
    have = cache_read(key)
    have.append(readouts)
    tmp = FOLD_CACHE / f"{key}.json.tmp"
    tmp.write_text(json.dumps(have[:12]))
    tmp.replace(FOLD_CACHE / f"{key}.json")


def fold_batch(pairs, job_dir, mode, msa_path, budget_per_fold, progress=None):
    """Fold a list of (name, receptor, peptide) in ONE process. {name: readouts}.

    Every fold in the list goes into a single input directory and a single
    predict call, because the cost is almost all fixed: measured on this machine
    a lone fold takes 53.9s and four take 74.7s, which puts model construction at
    ~47s and the marginal fold at ~6.9s. Splitting a job across two calls pays
    that 47s twice.

    Progress therefore cannot come from counting calls. It is read off the
    prediction directory instead, which boltz fills in as it goes.
    """
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
    res = job_dir / f"boltz_results_{inputs.name}" / "predictions"
    stop = threading.Event()

    def watch():
        while not stop.wait(3.0):
            try:
                n = sum(1 for d in res.iterdir()
                        if (d / f"{d.name}_model_0.pdb").exists())
            except OSError:
                n = 0
            progress(n)

    watcher = None
    if progress is not None:
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    try:
        if os.environ.get("BOLTZ_NO_KERNELS") and "--no_kernels" not in cmd:
            cmd.append("--no_kernels")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=max(900, 300 + len(pairs) * budget_per_fold * 3))
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=5)
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout + proc.stderr)[-1500:])

    parser = PDBParser(QUIET=True)
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
    # With few scrambles the null has almost no degrees of freedom, and the
    # critical value explodes: at df = 2 a candidate needs t > 2.92 to reach
    # p < 0.05, so a real binder with a +11 margin still reads as nothing. That
    # is a statement about the job's size, not about the candidate, and the tool
    # says which rather than letting the user read a null result as evidence.
    job["power_note"] = None
    if pooled_df is not None and 0 < pooled_df < 5:
        from scipy import stats
        crit = float(stats.t.isf(0.05, pooled_df))
        job["power_note"] = (
            f"The scramble null has only {pooled_df} degree(s) of freedom, so a "
            f"candidate needs t > {crit:.2f} before this job can call it a hit. "
            f"An 'indistinguishable' verdict here may mean the job is too small "
            f"rather than the candidate inactive — raise scrambles, or screen "
            f"more candidates together.")
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
        # The pooled spread is the denominator, and because it is estimated from
        # a handful of deviations the ratio is a t statistic on pooled_df, not a
        # z. Treating it as a z is what let a GSGSGSGSGSGS linker -- which binds
        # nothing -- pass a fixed |z| >= 1 cutoff at 1.06. On the t scale that
        # same candidate is p = 0.14, which is the honest answer.
        t = p = None
        if pooled and pooled > 1e-6:
            se = pooled * float(np.sqrt(1.0 / len(got) + 1.0 / len(nulls)))
            if se > 1e-9:
                t = (v - nm) / se
                from scipy import stats
                p = float(stats.t.sf(t, pooled_df))
        if t is None:
            verdict = (f"not enough scrambles to estimate a null "
                       f"(margin {v - nm:+.2f})")
        elif p < 0.01:
            verdict = "beats its own scrambles"
        elif p < 0.05:
            verdict = "suggestive — worth a replicate"
        else:
            verdict = "indistinguishable from its own scrambles"
        rows.append({**cand, "status": "scored", "score": v,
                     "replicate_sd": spread, "null_mean": nm, "null_sd": ns,
                     "margin": v - nm, "t_vs_own_scrambles": t, "p_one_sided": p,
                     "pooled_null_sd": pooled, "pooled_df": pooled_df,
                     "n_reps": len(got), "n_null": len(nulls),
                     "verdict": verdict})
    scored = [r for r in rows if r["status"] == "scored"]
    # rank on the t statistic where it exists, else on the raw margin
    scored.sort(key=lambda r: -(r["t_vs_own_scrambles"]
                                if r["t_vs_own_scrambles"] is not None
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

        # every candidate gets its own permutations; that is the whole point.
        # A scramble is folded once: the null asks what this composition scores
        # over orderings, so a fixed budget is better spent on more distinct
        # permutations than on re-folding a few of them.
        wanted, index = [], {}
        for ci, cand in enumerate(job["candidates"]):
            if cand.get("problems"):
                continue
            for rep in range(job["replicates"]):
                n = f"c{ci}_r{rep}"
                wanted.append((n, cand["peptide"]))
                index[n] = (ci, "reps", rep)
            for si, sc in enumerate(cand["scrambles"]):
                n = f"c{ci}_s{si}"
                wanted.append((n, sc))
                index[n] = (ci, "null_reps", si)

        with JOBS_LOCK:
            job["folds_total"] = len(wanted)

        # Draw what earlier runs already folded under these exact settings. Each
        # cached entry is one independent unseeded fold, so a replicate may take
        # one -- but a key is drawn from only as many times as it has distinct
        # entries, never reused twice inside a job.
        taken, pairs, reused = {}, [], 0
        for n, pep in wanted:
            key = fold_key(job["receptor"], pep, cfg)
            have = cache_read(key)
            k = taken.get(key, 0)
            if k < len(have):
                taken[key] = k + 1
                ci, kind, slot = index[n]
                cand = job["candidates"][ci]
                val = have[k].get(job["metric"])
                (cand["reps"] if kind == "reps"
                 else cand["null_reps"][slot]).append(val)
                reused += 1
            else:
                taken[key] = k + 1
                pairs.append((n, job["receptor"], pep))

        with JOBS_LOCK:
            job["reused"] = reused
        if pairs:
            step(f"folding {len(pairs)} structures "
                 f"({cfg['label'].lower()})" +
                 (f", {reused} reused" if reused else "") + " …", reused)
            got = fold_batch(
                pairs, job_dir, job["mode"], msa, cfg["sec_per_fold"],
                progress=lambda n: step(
                    f"folded {reused + n} of {len(wanted)}", reused + n))
            for name, _, pep in pairs:
                if name not in got:
                    continue
                ci, kind, slot = index[name]
                cand = job["candidates"][ci]
                val = got[name].get(job["metric"])
                (cand["reps"] if kind == "reps"
                 else cand["null_reps"][slot]).append(val)
                cache_append(fold_key(job["receptor"], pep, cfg), got[name])
            shutil.rmtree(job_dir / "boltz_results_inputs", ignore_errors=True)
        step(f"folded {len(wanted)} of {len(wanted)}", len(wanted))

        results, metric = summarise(job)
        with JOBS_LOCK:
            job["results"] = results
            job["state"] = "done"
            job["message"] = (f"complete — {job['folds_total']} folds"
                              + (f", {job.get('reused', 0)} reused from cache"
                                 if job.get("reused") else ""))
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

    @staticmethod
    def _example_view(ex):
        need, have = example_readiness(ex)
        left = need - have
        cfg = MODES[ex["mode"]]
        return {**ex, "folds": need, "cached": have,
                "eta_seconds": 0 if not left
                               else int(FIXED_STARTUP_SEC + left * cfg["sec_per_fold"])}

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
                          for k, v in MODES.items()},
                "examples": [self._example_view(e) for e in EXAMPLES]}))
        if self.path.startswith("/api/job/"):
            jid = self.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(jid)
                if not job:
                    return self._send(404, json.dumps({"error": "no such job"}))
                view = {k: job[k] for k in
                        ("state", "message", "folds_done", "folds_total", "mode",
                         "replicates", "metric", "msa", "reused", "power_note", "pooled_null_sd",
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
        nulls = max(1, min(6, int(req.get("scrambles", 3))))
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

        cands = []
        for p in peptides:
            probs = check_peptide(p)
            cands.append({"peptide": p, "problems": probs,
                          "scrambles": [] if probs else scrambles_of(p, nulls),
                          "reps": [], "null_reps": [[] for _ in range(nulls)]})

        jid = uuid.uuid4().hex[:12]
        n_folds = sum(reps + nulls for c in cands if not c["problems"])
        with JOBS_LOCK:
            JOBS[jid] = {"id": jid, "state": "queued", "message": "queued",
                         "receptor": receptor, "candidates": cands, "mode": mode,
                         "replicates": reps, "metric": metric,
                         "folds_total": n_folds, "folds_done": 0,
                         "started_at": time.time()}
        WORKQ.put(jid)
        return self._send(200, json.dumps({
            "job_id": jid, "folds": n_folds,
            "eta_seconds": int(FIXED_STARTUP_SEC
                                + n_folds * MODES[mode]["sec_per_fold"])}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    srv = HTTPServer((args.host, args.port), Handler)
    print(f"Peptide screening tool  →  http://{args.host}:{args.port}")
    print(f"  ~{FIXED_STARTUP_SEC}s to start, then "
          f"{MODES['quick']['sec_per_fold']}s/fold quick, "
          f"{MODES['careful']['sec_per_fold']}s/fold careful")
    print("  folds are cached, so re-screening a candidate is free")
    print("  every candidate is folded against permutations of itself")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
