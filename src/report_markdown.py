r"""Shared markdown helpers for the report compilers.

Python-Markdown applies emphasis parsing inside ``$...$``. Where an underscore
follows a non-word character -- ``\hat{x}_{t+1}``, ``\mathcal{L}_{\text{SimPO}}``
-- it opens an emphasis run, so the underscores are consumed and the delimiters
replaced with ``<em>``. MathJax then receives ``\hat{x}{t+1}`` and cannot parse
it. Underscores that follow word characters (``c_{\text{target}}``) are left
alone, which is why the breakage looks arbitrary from section to section.

Shielding the math spans before conversion and restoring them afterwards fixes
it without pulling in pymdownx.arithmatex for two documents.
"""

import re

# Display ($$...$$) first so inline matching never splits one in half.
_MATH_PATTERN = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.DOTALL)
_MATH_TOKEN = "zzmathspan{}zz"


def protect_math(text: str):
    """Replace math spans with inert tokens. Returns (text, spans).

    The tokens are lowercase alphanumeric, so no markdown construct touches
    them.
    """
    spans: list[str] = []

    def _stash(match: "re.Match[str]") -> str:
        spans.append(match.group(0))
        return _MATH_TOKEN.format(len(spans) - 1)

    return _MATH_PATTERN.sub(_stash, text), spans


def restore_math(html: str, spans: list) -> str:
    """Substitute the stashed math back in, failing loudly if one went missing."""
    for index, expression in enumerate(spans):
        token = _MATH_TOKEN.format(index)
        if token not in html:
            raise RuntimeError(f"math placeholder {token} lost during conversion")
        html = html.replace(token, expression)
    return html


# Loaded by the HTML templates. MathJax typesets asynchronously, so the Chrome
# invocation must also pass --virtual-time-budget or the PDF prints before any
# of this runs and every formula comes out as a blank gap.
MATHJAX_HEAD = """
    <script>
    window.MathJax = {
      tex: {inlineMath: [['$', '$']], displayMath: [['$$', '$$']]},
      options: {skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']}
    };
    </script>
    <script id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
