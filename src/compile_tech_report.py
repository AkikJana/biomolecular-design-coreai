import os
import subprocess
import markdown

def convert_tech_report_to_pdf(md_path: str, pdf_path: str):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Technical report markdown not found at: {md_path}")
        
    # Read the markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 1. Structural Enhancement: Replace the markdown metadata header with a clean arXiv Title Block
    # Find the title section (between start and first '---')
    parts = md_content.split("---", 1)
    if len(parts) > 1:
        body_content = parts[1]
    else:
        body_content = md_content

    arxiv_title_html = """
    <div class="arxiv-header">TECHNICAL REPORT</div>
    <div class="arxiv-title">Boltz-Fast: A Unified Framework for Memory-Efficient and Accelerated Biomolecular Design</div>
    <div class="arxiv-authors">
        <span class="author-name">Akik Jana</span><sup>1</sup>, 
        <span class="author-name">Dr. Arnab Bandyopadhyay</span><sup>2</sup>
    </div>
    <div class="arxiv-affiliations">
        <sup>1</sup>Work Integrated Learning Programmes (WILP), Birla Institute of Technology & Science, Pilani<br>
        <sup>2</sup>RnD Division, Dr. Reddy's Laboratories, Hyderabad
    </div>
    <div class="arxiv-date">June 15, 2026</div>
    <div class="arxiv-line-divider"></div>
    """

    # Replace the flow matching diagram if present or add custom style classes
    # Convert markdown body to HTML
    html_body = markdown.markdown(body_content, extensions=['tables', 'fenced_code'])
    
    # Custom stylesheet tailored for LaTeX/arXiv typography
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Boltz-Fast Technical Report</title>
    <!-- MathJax Setup for LaTeX Rendering -->
    <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
    </script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm;
            @bottom-right {{
                content: counter(page);
                font-family: "Times New Roman", Times, serif;
                font-size: 10pt;
            }}
        }}
        body {{
            font-family: "Times New Roman", Times, serif;
            font-size: 10.5pt;
            line-height: 1.5;
            color: #111111;
            margin: 0;
            padding: 0;
        }}
        
        /* --- arXiv TITLE BLOCK STYLING --- */
        .arxiv-header {{
            font-family: Arial, sans-serif;
            font-size: 8.5pt;
            font-weight: bold;
            letter-spacing: 1.5px;
            color: #555555;
            text-align: center;
            margin-bottom: 20px;
        }}
        .arxiv-title {{
            font-size: 19pt;
            font-weight: bold;
            text-align: center;
            line-height: 1.25;
            max-width: 680px;
            margin: 0 auto 15px auto;
            color: #000000;
        }}
        .arxiv-authors {{
            font-size: 11pt;
            text-align: center;
            margin-bottom: 5px;
            font-weight: normal;
        }}
        .author-name {{
            font-weight: 500;
        }}
        .arxiv-affiliations {{
            font-size: 9pt;
            text-align: center;
            color: #444444;
            line-height: 1.4;
            margin-bottom: 12px;
        }}
        .arxiv-date {{
            font-size: 10pt;
            text-align: center;
            color: #555555;
            margin-bottom: 25px;
        }}
        .arxiv-line-divider {{
            width: 100%;
            height: 1px;
            background-color: #cccccc;
            margin-bottom: 30px;
        }}
        
        /* --- REPORT TEXT STYLING --- */
        h1 {{
            font-size: 13pt;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #000000;
            margin-top: 30px;
            margin-bottom: 12px;
            border-bottom: 1px solid #111111;
            padding-bottom: 3px;
        }}
        h2 {{
            font-size: 11.5pt;
            font-weight: bold;
            color: #000000;
            margin-top: 22px;
            margin-bottom: 8px;
        }}
        h3 {{
            font-size: 10.5pt;
            font-weight: bold;
            font-style: italic;
            color: #222222;
            margin-top: 15px;
            margin-bottom: 6px;
        }}
        p {{
            margin-top: 0;
            margin-bottom: 12px;
            text-align: justify;
            text-indent: 1.5em; /* standard academic paragraph indent */
        }}
        p:first-of-type {{
            text-indent: 0; /* no indent on first paragraph of section */
        }}
        .abstract-container {{
            margin: 0 auto 30px auto;
            max-width: 600px;
            font-size: 9.5pt;
            line-height: 1.45;
            text-align: justify;
        }}
        .abstract-title {{
            font-weight: bold;
            text-align: center;
            text-transform: uppercase;
            font-size: 9.5pt;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        ul, ol {{
            margin-top: 0;
            margin-bottom: 12px;
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #dddddd;
            padding: 6px 10px;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
            color: #000000;
            font-weight: bold;
            border-bottom: 2px solid #aaaaaa;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        code {{
            font-family: monospace;
            background-color: #f4f4f4;
            padding: 1px 3px;
            border-radius: 2px;
            font-size: 8.5pt;
        }}
        pre {{
            background-color: #f9f9f9;
            padding: 10px;
            border: 1px solid #dddddd;
            border-radius: 4px;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        
        /* Page break controls */
        h1, h2, h3 {{
            page-break-after: avoid;
        }}
    </style>
</head>
<body>
    <div class="arxiv-title-block">
        {arxiv_title_html}
    </div>
    <div class="report-content">
        {html_body}
    </div>
</body>
</html>
"""
    
    # We want to format the Abstract specifically with the abstract container
    # Let's parse out the Abstract block in html_body and replace it with styled container
    abstract_prefix = "<h2>Abstract</h2>"
    if abstract_prefix in html_body:
        html_body = html_body.replace(abstract_prefix, "")
        # Find the paragraph after abstract_prefix
        # In Markdown, it will be <p>Abstract text...</p>
        parts = html_body.split("<p>", 1)
        if len(parts) > 1:
            p_parts = parts[1].split("</p>", 1)
            abstract_text = p_parts[0]
            rest_of_body = p_parts[1]
            
            styled_abstract = f"""
            <div class="abstract-container">
                <div class="abstract-title">Abstract</div>
                <p style="text-indent: 0;">{abstract_text}</p>
            </div>
            """
            html_body = styled_abstract + rest_of_body

    # Re-inject the updated html_body
    html_content = html_content.replace("{html_body}", html_body)

    # Save the styled HTML to a temporary file
    temp_html_path = "/tmp/boltz_fast_technical_report.html"
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"[PDF] Styled HTML generated at: {temp_html_path}")
    
    # Locate Chrome executable on Mac
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome_path):
        raise FileNotFoundError("Google Chrome application not found at standard Mac path.")
        
    # Run Chrome in headless print-to-pdf mode
    print("[PDF] Running Headless Chrome to compile PDF...")
    command = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--timeout=5000",
        f"--print-to-pdf={pdf_path}",
        temp_html_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Chrome PDF generation failed: {result.stderr}")
        
    print(f"[PDF] Successfully generated PDF at: {pdf_path}")

if __name__ == "__main__":
    brain_dir = "/Users/akikjana/.gemini/antigravity-cli/brain/4f6026cb-893e-48e8-91fe-c87b3988df92"
    md_file = os.path.join(brain_dir, "boltz_fast_technical_report.md")
    pdf_file = "/Users/akikjana/Documents/BiomolecularDesign/boltz_fast_technical_report.pdf"
    
    convert_tech_report_to_pdf(md_file, pdf_file)
