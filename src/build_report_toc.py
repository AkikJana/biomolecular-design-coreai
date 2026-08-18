"""Build the report's Table of Contents from the rendered PDF.

Page numbers in a contents page have to be true, and the only way to know where
a heading actually lands is to render and look. Inserting the contents then
changes pagination, so this iterates render → read → rewrite until the page
numbers stop moving, and fails if they never settle.

Numbering follows the guidelines: front matter in lower-case roman, body in
arabic with Chapter 1 on page 1, matching number_report_pages.py.

Usage:
    python src/build_report_toc.py
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
MD = REPO_ROOT / "reports" / "final_dissertation_report.md"
PDF = REPO_ROOT / "final_dissertation_report.pdf"
MARKER = "<!--TOC-->"

# A phrase from the Introduction's opening sentence rather than its heading.
# The heading also appears in the Table of Contents, which is printed *before*
# the Introduction -- keying on it made the contents page itself register as
# body page 1, and every contents entry then resolved to "1".
BODY_MARKER = "moved from a research problem to an engineering one"


def render():
    for script in ("compile_pdf.py", "number_report_pages.py"):
        r = subprocess.run([sys.executable, str(REPO_ROOT / "src" / script)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{script} failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")


def page_texts():
    return [(p.extract_text() or "") for p in PdfReader(str(PDF)).pages]


def _roman(n):
    vals = ((100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"),
            (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s); n -= v
    return "".join(out)


def locate(pages, needles):
    """Printed page label for each needle.

    Front matter is lower-case roman and the body is arabic with Chapter 1 on
    page 1, matching what number_report_pages.py stamps -- a contents page that
    disagreed with the printed folio would be worse than none.
    """
    body = next(i for i, t in enumerate(pages)
                if BODY_MARKER in " ".join(t.split()))
    found = {}

    # The body is searched first and the front matter only for what the body
    # does not contain. Skipping pages that carry a "LIST OF TABLES" header is
    # not enough: with fifty-two entries that list runs over several pages and
    # only the first is marked, so continuation pages look like ordinary text
    # and every caption on them resolves to the contents page instead of to its
    # own table. Front matter still resolves CERTIFICATE and the rest to roman
    # numerals, because nothing in the body matches those.
    for lo, hi in ((body, len(pages)), (0, body)):
        for i in range(lo, hi):
            flat = " ".join(pages[i].split())
            if any(k in flat for k in ("TABLE OF CONTENTS", "LIST OF FIGURES",
                                       "LIST OF TABLES")):
                continue
            for n in needles:
                if n in found:
                    continue
                if " ".join(n.split()) in flat:
                    found[n] = _roman(i) if i < body else str(i - body + 1)
    return found


def headings():
    """(level, display text, lookup key) for everything the contents lists."""
    md = MD.read_text()
    out = []
    for line in md.splitlines():
        if line.startswith("# ") and not line.startswith("# BITS"):
            title = line[2:].strip()
            if title in ("A REPORT", "TABLE OF CONTENTS"):
                continue
            out.append((1, title, title))
        elif re.match(r"^## \d+\.\d+ ", line):
            title = line[3:].strip()
            out.append((2, title, title))
        elif re.match(r"^### \d+\.\d+ ", line):
            title = line[4:].strip()
            out.append((2, title, title))
    return out


def captions(prefix):
    """Every `Figure N:` / `Table N:` caption, in document order.

    Two forms are accepted. Figures caption with a `####` heading; tables use a
    bold paragraph, because a heading-style table caption inside Section 7 would
    sit at the same level as the 7.x subsections and read as structure rather
    than as a caption. Matching only the heading form is how the list of tables
    came to have four entries for fifty-two tables.
    """
    md = MD.read_text()
    pat = rf"^(?:#{{2,4}} |\*\*)({prefix} \d+: .+?)(?:\*\*)?$"
    return [m.group(1).strip() for m in re.finditer(pat, md, re.M)]


def dots(text, page, width=78):
    page_s = str(page)
    fill = max(3, width - len(text) - len(page_s))
    return f"{text} {'.' * fill} {page_s}"


def build(pages):
    heads = headings()
    figs = captions("Figure")
    tabs = captions("Table")
    pos = locate(pages, [h[2] for h in heads] + figs + tabs)

    # Long headings wrap and then get stretched by the justified body style,
    # so the contents uses a shortened label where the full heading will not fit.
    short = {"CHECKLIST OF ITEMS FOR THE FINAL DISSERTATION / PROJECT / "
             "PROJECT WORK REPORT": "CHECKLIST OF ITEMS FOR THE FINAL REPORT",
             "APPENDIX B — REPRODUCTION OF THE REPORTED RESULTS":
                 "APPENDIX B — REPRODUCTION OF RESULTS",
             "7. MEASURED RESULTS AGAINST PRETRAINED WEIGHTS":
                 "7. MEASURED RESULTS VS PRETRAINED WEIGHTS",
             "7.1 The Low-Rank Pair Representation Is Not Reachable on "
             "Pretrained Weights": "7.1 Low-Rank Pair Representation on "
             "Pretrained Weights",
             "7.5 Measurement Reproducibility: Single Folds Do Not Reproduce":
                 "7.5 Measurement Reproducibility"}

    lines = ["# TABLE OF CONTENTS", ""]
    for level, title, key in heads:
        page = pos.get(key)
        if page is None:
            continue
        indent = "" if level == 1 else "&nbsp;&nbsp;&nbsp;&nbsp;"
        lines.append(f"{indent}{dots(short.get(title, title), page)}  ")
    lines += ["", "&nbsp;", "", "## LIST OF FIGURES", ""]
    for f in figs:
        if f in pos:
            lines.append(f"{dots(f, pos[f])}  ")
    lines += ["", "&nbsp;", "", "## LIST OF TABLES", ""]
    for t in tabs:
        if t in pos:
            lines.append(f"{dots(t, pos[t])}  ")
    lines += ["", '<div class="page-break"></div>']
    return "\n".join(lines), pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=5)
    args = ap.parse_args()

    md = MD.read_text()
    if MARKER not in md:
        # replace an existing contents block so the script is re-runnable
        start = md.index("# TABLE OF CONTENTS")
        end = md.index("# 1. INTRODUCTION")
        md = md[:start] + MARKER + "\n\n" + md[end:]
        MD.write_text(md)

    prev = None
    for it in range(1, args.max_iters + 1):
        render()
        toc, pos = build(page_texts())
        if pos == prev:
            print(f"page numbers stable after {it} iteration(s)")
            break
        md = MD.read_text()
        md = (md.replace(MARKER, toc) if MARKER in md else
              md[:md.index("# TABLE OF CONTENTS")] + toc + "\n\n"
              + md[md.index("# 1. INTRODUCTION"):])
        MD.write_text(md)
        prev = pos
    else:
        raise SystemExit("contents page numbers did not converge")

    render()
    print(f"contents built: {len(pos)} entries")


if __name__ == "__main__":
    main()
