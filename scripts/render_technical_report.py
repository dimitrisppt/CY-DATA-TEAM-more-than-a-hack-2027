# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown-it-py==4.0.0", "weasyprint==66.0", "PyMuPDF==1.26.6"]
# ///
"""Render the team report locally: uv run --no-project scripts/render_technical_report.py."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import fitz
from markdown_it import MarkdownIt
from weasyprint import HTML

REPO = Path(__file__).resolve().parents[1]
AUTHORS = ("Dimitris Papatheodoulou", "Evripides Alevras", "Andreas Petridis")
TITLE = "CY-DATA-TEAM: Football possession from incomplete tracking"
BASE = "https://github.com/dimitrisppt/CY-DATA-TEAM-more-than-a-hack-2027/blob/main/docs/"
CSS = """
@page {
  size: A4; margin: 18mm 18mm 18mm;
  @top-left { content: "CY-DATA-TEAM  /  TECHNICAL REPORT";
    font: 7.3pt 'DejaVu Sans'; color: #004d98; letter-spacing: .6pt; }
  @top-right { content: "17 NOVEMBER 2026";
    font: 7.3pt 'DejaVu Sans'; color: #596b7c; }
  @bottom-left { content: "More Than a Hack 2027  ·  v7 GPU-tuned submission";
    font: 7.1pt 'DejaVu Sans'; color: #596b7c; }
  @bottom-right { content: counter(page) " / " counter(pages);
    font: 8pt 'DejaVu Sans'; color: #a50044; }
}
@page:first { @top-left { content: none; } @top-right { content: none; } }
body { font: 9.65pt/1.46 'DejaVu Sans', sans-serif; color: #25324a; margin: 0; }
section { break-before: page; }
section:first-of-type { break-before: auto; }
.eyebrow { font-size: 8.6pt; text-transform: uppercase; letter-spacing: 1pt;
  color: #a50044; font-weight: bold; padding-bottom: 4mm; border-top: 3pt solid transparent;
  padding-top: 4mm; position: relative; }
.eyebrow::before { content: ""; position: absolute; top: -3pt; left: 0; right: 0;
  height: 3pt; background: repeating-linear-gradient(90deg,
    #004d98 0%, #004d98 20%, #a50044 20%, #a50044 40%); }
h1 { color: #004d98; font-size: 29pt; line-height: 1.12; margin: 1mm 0 3mm;
  letter-spacing: -.8pt; bookmark-level: none; }
.subtitle { color: #596b7c; font-size: 10.3pt; margin: 0 0 5mm; }
h2 { color: #004d98; font-size: 17pt; line-height: 1.2; margin: 0 0 5mm;
  border-bottom: .6pt solid #edbb00; padding-bottom: 3mm; bookmark-level: 1; }
section:first-of-type h2 { margin-top: 7mm; font-size: 15pt; }
h3 { color: #a50044; font-size: 10.6pt; line-height: 1.25; margin: 4mm 0 2mm;
  bookmark-level: 2; }
h1, h2, h3 { break-after: avoid; }
p { margin: 0 0 2.9mm; orphans: 3; widows: 3; }
strong { font-weight: bold; color: #17305a; }
a { color: #004d98; text-decoration: underline; text-decoration-thickness: .4pt; }
blockquote { margin: 3mm 0 4mm; padding: 3mm 3.5mm; background: #eff4fb;
  border-left: 2.5pt solid #a50044; font-size: 9pt; break-inside: avoid; }
blockquote p { margin: 0; }
table { width: 100%; border-collapse: collapse; margin: 3mm 0 4mm;
  font-size: 8.5pt; line-height: 1.38; }
thead { display: table-header-group; background: #a50044; color: white; }
th { text-align: left; font-weight: bold; padding: 2.2mm; }
th:first-child { background: #004d98; }
td { padding: 2.2mm; border-bottom: .5pt solid #d7e1f0; vertical-align: top; }
tbody tr:nth-child(even) { background: #f0f4fa; }
tr { break-inside: avoid; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: .9em; }
pre { padding: 2.8mm; border: .5pt solid #d7e1f0; background: #f3f5fa;
  font-size: 8pt; line-height: 1.4; white-space: pre-wrap; overflow-wrap: anywhere;
  break-inside: avoid; margin: 3mm 0; }
ol { margin: 2mm 0 0; padding-left: 5mm; font-size: 7.9pt; line-height: 1.35; }
li { margin-bottom: 1.6mm; break-inside: avoid; }
li p { margin: 0; }
section:last-of-type { font-size: 9.1pt; line-height: 1.4; }
section:last-of-type blockquote { font-size: 8.5pt; background: #faf0f4; }
"""


def render_markdown(source: str) -> str:
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    sections = []
    for section in source.split("<!-- pagebreak -->"):
        tokens = parser.parse(section)
        for index, token in enumerate(tokens):
            if token.type == "heading_open":
                label = tokens[index + 1].content.lower()
                token.attrSet("id", re.sub(r"[^\w\- ]", "", label).replace(" ", "-"))
            for child in token.children or []:
                if child.type == "link_open":
                    target = child.attrGet("href") or ""
                    if not target.startswith(("https://", "http://", "#", "mailto:")):
                        child.attrSet("href", urljoin(BASE, target))
        sections.append("<section>" + parser.renderer.render(tokens, parser.options, {}) + "</section>")
    meta = "".join(f'<meta name="author" content="{html.escape(name)}">' for name in AUTHORS)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">{meta}'
            f'<title>{html.escape(TITLE)}</title><style>{CSS}</style></head>'
            f'<body>{"".join(sections)}</body></html>')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "docs/CY-DATA-TEAM.md")
    parser.add_argument("--output", type=Path, default=REPO / "submission/CY-DATA-TEAM.pdf")
    parser.add_argument("--preview-dir", type=Path, default=REPO / "outputs/technical-report")
    args = parser.parse_args()
    source = args.source.read_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    document = render_markdown(source)
    (args.preview_dir / "report.html").write_text(document)
    HTML(string=document, base_url=str(REPO)).write_pdf(args.output)
    with fitz.open(args.output) as pdf:
        text = "\n".join(page.get_text() for page in pdf)
        for phrase in (*AUTHORS, "0.629368", "0.594200", "v7", "199,319"):
            if phrase not in text:
                raise ValueError(f"Required report text missing from PDF: {phrase}")
        (args.preview_dir / "report.txt").write_text(text)
        for index, page in enumerate(pdf, 1):
            page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(
                args.preview_dir / f"page-{index:02d}.png")
        report = {"output": str(args.output), "pages": len(pdf),
                  "bytes": args.output.stat().st_size,
                  "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "page_word_counts": [len(page.get_text().split()) for page in pdf],
                  "authors": pdf.metadata.get("author"),
                  "links": sum(len(page.get_links()) for page in pdf)}
    (args.preview_dir / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
