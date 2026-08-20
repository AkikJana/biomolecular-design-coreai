"""Render the preprint to PDF in a plain academic style.

Deliberately separate from compile_pdf.py: that one stamps a BITS cover page,
roman front matter and WILP page numbering onto every document it touches,
none of which belongs on a preprint.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

import base64

import markdown

REPO_ROOT = Path(__file__).resolve().parent.parent

CSS = """
@page { size: A4; margin: 22mm 20mm; }
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 10.5pt; line-height: 1.45; color: #000;
    max-width: 100%; margin: 0;
}
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 4pt 0; text-align: left; }
h2 { font-size: 12pt; margin: 16pt 0 5pt 0; border-bottom: 0.5pt solid #999;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }
p  { margin: 0 0 6pt 0; text-align: justify; }
/* The author block and affiliation sit between the title and the first rule. */
h1 + p, h1 + p + p, h1 + p + p + p { text-align: left; margin-bottom: 2pt; }
table {
    border-collapse: collapse; width: 100%; margin: 8pt 0 10pt 0;
    font-size: 9pt; page-break-inside: avoid;
}
th, td { border: 0.5pt solid #666; padding: 3pt 5pt; }
th { background: #eee; font-weight: bold; }
blockquote {
    margin: 8pt 0 8pt 14pt; padding-left: 10pt;
    border-left: 2pt solid #bbb; font-style: italic;
}
img { max-width: 100%; display: block; margin: 8pt auto 3pt auto; }
.figcap { font-size: 9pt; text-align: left; margin: 0 0 10pt 0; }
code { font-family: Menlo, Consolas, monospace; font-size: 9pt; }
hr { border: none; border-top: 0.5pt solid #bbb; margin: 12pt 0; }
/* Reference list: hanging indent, and never split an entry across pages. */
h2#references + ol, h2:last-of-type + ol { font-size: 9.5pt; }
ol li, ul li { margin-bottom: 3pt; page-break-inside: avoid; }
"""


def embed_images(html: str, base: Path) -> str:
    """Inline <img> sources as data URIs.

    The HTML is rendered from a temp file elsewhere on disk, so relative
    image paths in the markdown would otherwise resolve to nothing.
    """
    def repl(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith(("data:", "http:", "https:")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.exists():
            print(f"  warning: image not found: {src}")
            return m.group(0)
        uri = base64.b64encode(path.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/png;base64,{uri}")

    return re.sub(r'<img[^>]*src="([^"]+)"', repl, html)


def convert(md_path: Path, pdf_path: Path) -> None:
    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html_body = embed_images(html_body, md_path.parent)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome):
        raise FileNotFoundError(f"Chrome not found at {chrome}")

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(html)
        tmp = fh.name
    try:
        result = subprocess.run(
            [chrome, "--headless", "--disable-gpu",
             "--run-all-compositor-stages-before-draw",
             f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer", tmp],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Chrome failed: {result.stderr}")
    finally:
        os.unlink(tmp)
    print(f"[PDF] wrote {pdf_path}")


if __name__ == "__main__":
    convert(REPO_ROOT / "reports" / "preprint_scramble_control.md",
            REPO_ROOT / "preprint_scramble_control.pdf")
