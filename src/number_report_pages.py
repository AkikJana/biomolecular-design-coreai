"""Stamp WILP-conformant page numbers onto the rendered report.

The guidelines require roman numerals for everything preceding the Introduction
and arabic numerals from the Introduction onwards, with Chapter 1 on page 1.
Chrome's print-to-PDF numbers pages continuously from the cover and offers no
way to restart a counter, so numbering is applied here instead --
compile_pdf.py passes --no-pdf-header-footer and this module stamps the numbers.

The cover carries no number, by convention.

Usage:
    python src/number_report_pages.py --pdf final_dissertation_report.pdf
"""

import argparse
import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

REPO_ROOT = Path(__file__).resolve().parents[1]
# A phrase from the Introduction's opening sentence rather than its heading.
# The heading also appears in the Table of Contents, which is printed *before*
# the Introduction -- keying on it made the contents page itself register as
# body page 1, and every contents entry then resolved to "1".
BODY_MARKER = "moved from a research problem to an engineering one"


def roman(n):
    vals = ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
            (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
            (5, "v"), (4, "iv"), (1, "i"))
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s); n -= v
    return "".join(out)


def find_body_start(reader):
    """Index of the page on which the Introduction begins."""
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "")
        if BODY_MARKER in " ".join(text.split()):
            return i
    raise SystemExit(f"could not locate {BODY_MARKER!r}; cannot number pages")


def stamp(width, height, label):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))
    c.setFont("Times-Roman", 10)
    c.setFillGray(0.25)
    c.drawCentredString(width / 2.0, 0.55 * 72, label)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(REPO_ROOT / "final_dissertation_report.pdf"))
    args = ap.parse_args()

    reader = PdfReader(args.pdf)
    body = find_body_start(reader)
    writer = PdfWriter()

    for i, page in enumerate(reader.pages):
        box = page.mediabox
        w, h = float(box.width), float(box.height)
        if i == 0:
            label = None                       # cover carries no number
        elif i < body:
            label = roman(i)                   # i, ii, iii … for front matter
        else:
            label = str(i - body + 1)          # Chapter 1 starts at page 1
        if label:
            page.merge_page(stamp(w or A4[0], h or A4[1], label))
        writer.add_page(page)

    with open(args.pdf, "wb") as fh:
        writer.write(fh)
    print(f"numbered {len(reader.pages)} pages: "
          f"cover unnumbered, front matter i–{roman(body - 1)}, "
          f"body 1–{len(reader.pages) - body}")


if __name__ == "__main__":
    main()
