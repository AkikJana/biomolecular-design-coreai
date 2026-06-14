import os
import subprocess
import markdown

def convert_md_to_pdf(md_path: str, pdf_path: str):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown report not found at: {md_path}")
        
    # Read the markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 1. Structural Enhancement: Replace the markdown metadata header with a formal Cover Page
    # Find the cover block (between start and first '---')
    parts = md_content.split("---", 1)
    if len(parts) > 1:
        # We have a cover section to replace
        body_content = parts[1]
    else:
        body_content = md_content

    cover_page_html = """
    <div class="cover-page">
        <div class="cover-header">BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI</div>
        <div class="cover-subheader">WORK INTEGRATED LEARNING PROGRAMMES (WILP) DIVISION</div>
        <div class="cover-divider"></div>
        <div class="cover-title">Efficiency-Optimized Generative Paradigms for Large-Scale Biomolecular Design</div>
        <div class="cover-subtitle">Integrating Speculative Decoding and Preference Alignment</div>
        
        <div class="cover-meta-grid">
            <div class="meta-label">Course:</div>
            <div class="meta-val">AIMLCZG628T — Dissertation</div>
            
            <div class="meta-label">Submitted by:</div>
            <div class="meta-val">Akik Jana (BITS Student ID: 2024AB05287)</div>
            
            <div class="meta-label">Employing Organization:</div>
            <div class="meta-val">Mu Sigma, Bangalore</div>
            
            <div class="meta-label">Supervisor:</div>
            <div class="meta-val">Dr. Arnab Bandyopadhyay (Dr. Reddy's Laboratories)</div>
            
            <div class="meta-label">Additional Examiner:</div>
            <div class="meta-val">Dr. Ahana Das (Sanofi India)</div>
        </div>
        <div class="cover-footer">June 2026</div>
    </div>
    <div class="page-break"></div>
    """

    # 2. Structural Enhancement: Replace the raw Mermaid text block with a clean CSS Flow-Chart
    mermaid_block = """```mermaid
graph TD
    subgraph Structure Generation
        A[Binder Sequence] --> B[Speculative Flow Sampler]
        B -->|Fast Draft Model| C[Draft Coords]
        B -->|Full Target Model| D[Parallel Verification]
        D --> E[Verified PDB Complex]
    end
    
    subgraph Agentic Evaluation Loop
        E --> F[Agent Rosetta]
        F -->|Tool: evaluate_interface| G[Structural Feedback]
        G -->|H-Bonds, pLDDT, ipSAE| F
        F -->|Decision: Mutate Sequence| A
    end
    
    subgraph Model Alignment
        G --> H[Union Mask Clustering]
        H --> I[g-DPO Loss Module]
        I -->|Trajectory Gradient Updates| B
    end
```"""

    flowchart_html = """
    <div class="flowchart-container">
        <div class="flowchart-box primary">
            <h3>1. Structure Generation</h3>
            <p>Draft via cheap model; verify in parallel via Boltz-1/2 (AF3) solver.</p>
        </div>
        <div class="flowchart-arrow">➔</div>
        <div class="flowchart-box success">
            <h3>2. Agentic Loop</h3>
            <p>Agent Rosetta runs Design-Filter-Verify using structural feedback (H-Bonds, pLDDT, ipSAE).</p>
        </div>
        <div class="flowchart-arrow">➔</div>
        <div class="flowchart-box warning">
            <h3>3. Preference Alignment</h3>
            <p>Compute Union Mask clusters and execute g-DPO gradient updates to align folding.</p>
        </div>
    </div>
    """
    
    # Replace Mermaid block with the visual flowchart HTML
    body_content = body_content.replace(mermaid_block, flowchart_html)

    # Convert remaining markdown body to HTML
    html_body = markdown.markdown(body_content, extensions=['tables', 'fenced_code'])
    
    # 3. Apply Premium Academic CSS Stylesheet
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>M.Tech Dissertation Progress Report</title>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm;
            @bottom-right {{
                content: counter(page);
                font-family: Arial, sans-serif;
                font-size: 9pt;
            }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #2d3748;
            margin: 0;
            padding: 0;
        }}
        
        /* --- COVER PAGE STYLING --- */
        .cover-page {{
            height: 85vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-sizing: border-box;
            padding-top: 5cm;
        }}
        .cover-header {{
            font-size: 12pt;
            font-weight: bold;
            color: #1a365d;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        .cover-subheader {{
            font-size: 9pt;
            color: #718096;
            letter-spacing: 0.5px;
            margin-bottom: 40px;
        }}
        .cover-divider {{
            width: 80px;
            height: 3px;
            background-color: #3182ce;
            margin-bottom: 50px;
        }}
        .cover-title {{
            font-size: 20pt;
            font-weight: 800;
            color: #0c2340;
            line-height: 1.3;
            max-width: 650px;
            margin-bottom: 15px;
        }}
        .cover-subtitle {{
            font-size: 13pt;
            color: #4a5568;
            margin-bottom: 60px;
        }}
        .cover-meta-grid {{
            display: grid;
            grid-template-columns: 200px 1fr;
            gap: 12px;
            text-align: left;
            max-width: 550px;
            width: 100%;
            margin-bottom: 60px;
            font-size: 9.5pt;
            border-top: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
            padding: 20px 0;
        }}
        .meta-label {{
            font-weight: bold;
            color: #718096;
        }}
        .meta-val {{
            color: #2d3748;
        }}
        .cover-footer {{
            font-size: 9.5pt;
            color: #718096;
            margin-top: auto;
        }}
        .page-break {{
            page-break-before: always;
        }}
        
        /* --- FLOWCHART BOX STYLING --- */
        .flowchart-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 25px 0;
            padding: 15px;
            background-color: #f7fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            page-break-inside: avoid;
        }}
        .flowchart-box {{
            flex: 1;
            padding: 12px;
            background-color: #fff;
            border-radius: 6px;
            border: 1px solid #cbd5e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .flowchart-box.primary {{ border-top: 4px solid #3182ce; }}
        .flowchart-box.success {{ border-top: 4px solid #48bb78; }}
        .flowchart-box.warning {{ border-top: 4px solid #dd6b20; }}
        .flowchart-box h3 {{ 
            margin: 0 0 5px 0; 
            font-size: 9.5pt; 
            color: #1a365d; 
            border: none; 
            padding: 0; 
        }}
        .flowchart-box p {{ 
            margin: 0; 
            font-size: 8pt; 
            color: #4a5568; 
            line-height: 1.4;
            text-align: center;
        }}
        .flowchart-arrow {{
            padding: 0 10px;
            font-size: 16pt;
            color: #cbd5e0;
            font-weight: bold;
        }}
        
        /* --- REPORT TEXT STYLING --- */
        h1 {{
            font-size: 18pt;
            color: #0c2340;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #0c2340;
            padding-bottom: 5px;
        }}
        h2 {{
            font-size: 13pt;
            color: #1a365d;
            border-bottom: 1.5px solid #cbd5e0;
            padding-bottom: 4px;
            margin-top: 30px;
            margin-bottom: 12px;
        }}
        h3 {{
            font-size: 10.5pt;
            color: #2c5282;
            margin-top: 20px;
            margin-bottom: 8px;
        }}
        p {{
            margin-top: 0;
            margin-bottom: 12px;
            text-align: justify;
        }}
        ul, ol {{
            margin-top: 0;
            margin-bottom: 12px;
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 5px;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #cbd5e0;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #ebf8ff;
            color: #2b6cb0;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f7fafc;
        }}
        blockquote {{
            margin: 20px 0;
            padding: 12px 20px;
            background-color: #f7fafc;
            border-left: 4px solid #3182ce;
            color: #4a5568;
            font-style: italic;
            page-break-inside: avoid;
        }}
        code {{
            font-family: "SFMono-Regular", Consolas, Menlo, monospace;
            background-color: #f7fafc;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 8.5pt;
            border: 1px solid #e2e8f0;
        }}
        pre {{
            background-color: #f7fafc;
            padding: 12px;
            border: 1px solid #cbd5e0;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            border: none;
        }}
        img {{
            max-width: 90%;
            height: auto;
            display: block;
            margin: 25px auto;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            page-break-inside: avoid;
        }}
        
        /* Page break controls */
        h2, h3 {{
            page-break-after: avoid;
        }}
    </style>
</head>
<body>
    {cover_page_html}
    <div class="report-content">
        {html_body}
    </div>
</body>
</html>
"""
    
    # Save the styled HTML to a temporary file
    temp_html_path = "/tmp/mid_semester_report.html"
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
        f"--print-to-pdf={pdf_path}",
        temp_html_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Chrome PDF generation failed: {result.stderr}")
        
    print(f"[PDF] Successfully generated PDF at: {pdf_path}")

if __name__ == "__main__":
    brain_dir = "/Users/akikjana/.gemini/antigravity-cli/brain/4cdb7261-e55b-4efc-9ffa-c6509d76c9c2"
    md_file = os.path.join(brain_dir, "mid_semester_report.md")
    pdf_file = "/Users/akikjana/Documents/BiomolecularDesign/mid_semester_report.pdf"
    
    convert_md_to_pdf(md_file, pdf_file)
