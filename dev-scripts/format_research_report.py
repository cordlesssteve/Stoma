#!/usr/bin/env python3
"""
Format research reports from JSON to Markdown and PDF.
Converts your Ollama analysis results into beautiful, shareable formats.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stoma.reports.export_formats import MarkdownExporter, HTMLExporter
from stoma.reports.base_generator import GeneratedReport, ReportSection

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False


class ResearchReportFormatter:
    """Format research reports with citations into beautiful documents."""

    def __init__(self, output_dir: str = "formatted_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_json_report(self, json_file: str, formats: list = ["markdown", "html", "pdf"]):
        """Convert JSON research report to multiple formats."""

        print(f"📄 Formatting research report: {json_file}")

        # Load JSON data
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return {}

        # Extract data
        topic = data.get('topic', 'Research Analysis')
        analysis = data.get('analysis_with_citations', data.get('analysis', 'No analysis available'))
        model = data.get('model', 'Unknown')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        citations = data.get('citations_used', [])

        # Create base filename
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        base_filename = f"{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create GeneratedReport object
        report = GeneratedReport(
            report_id=base_filename,
            title=f"Research Analysis: {topic.title()}",
            template_id="research_analysis",
            sections=[
                ReportSection(
                    section_id="analysis",
                    title="Analysis",
                    content=analysis,
                    section_type="analysis",
                    order=2,
                    subsections=[]
                )
            ],
            metadata={
                "topic": topic,
                "model": model,
                "timestamp": timestamp,
                "citations_count": len(citations),
                "format": "research_analysis"
            },
            generated_at=datetime.fromisoformat(timestamp.replace('Z', '+00:00')) if 'T' in timestamp else datetime.now()
        )

        # Add citations section if available
        if citations:
            citations_content = self._format_citations_section(citations)
            report.sections.append(
                ReportSection(
                    section_id="references",
                    title="References",
                    content=citations_content,
                    section_type="data",
                    order=3,
                    subsections=[]
                )
            )

        # Add metadata section
        metadata_content = f"""**Research Details:**
- **Topic:** {topic}
- **Model:** {model}
- **Generated:** {timestamp}
- **Citations:** {len(citations)} sources
"""
        report.sections.insert(0, ReportSection(
            section_id="metadata",
            title="Metadata",
            content=metadata_content,
            section_type="summary",
            order=1,
            subsections=[]
        ))

        # Generate formats
        output_files = {}

        if "markdown" in formats:
            output_files["markdown"] = self._generate_markdown(report, base_filename)

        if "html" in formats:
            output_files["html"] = self._generate_html(report, base_filename)

        if "pdf" in formats and WEASYPRINT_AVAILABLE:
            output_files["pdf"] = self._generate_pdf(report, base_filename)
        elif "pdf" in formats:
            print("⚠️  PDF generation skipped - WeasyPrint not available")
            print("   Install with: pip install weasyprint")

        return output_files

    def _format_citations_section(self, citations):
        """Format citations into a proper bibliography section."""

        if not citations:
            return "No citations available."

        lines = ["## Bibliography\n"]

        for i, citation in enumerate(citations, 1):
            title = citation.get('title', 'Unknown Title')
            authors = citation.get('authors', ['Unknown Author'])
            published = citation.get('published', 'Unknown Date')
            url = citation.get('url', '#')

            # Format authors
            if isinstance(authors, list):
                author_str = ', '.join(authors) if authors else 'Unknown Author'
            else:
                author_str = str(authors)

            # Create citation entry
            lines.append(f"**[{i}]** {author_str}. *{title}*. {published}. [{url}]({url})")
            lines.append("")

        return "\n".join(lines)

    def _generate_markdown(self, report: GeneratedReport, base_filename: str) -> str:
        """Generate Markdown format."""

        try:
            exporter = MarkdownExporter()
            markdown_content = exporter.export(report)

            # Save to file
            output_file = self.output_dir / f"{base_filename}.md"
            with open(output_file, 'w') as f:
                f.write(markdown_content)

            print(f"✅ Markdown saved: {output_file}")
            return str(output_file)

        except Exception as e:
            print(f"❌ Markdown generation failed: {e}")
            return None

    def _generate_html(self, report: GeneratedReport, base_filename: str) -> str:
        """Generate HTML format."""

        try:
            exporter = HTMLExporter()
            html_content = exporter.export(report)

            # Save to file
            output_file = self.output_dir / f"{base_filename}.html"
            with open(output_file, 'w') as f:
                f.write(html_content)

            print(f"✅ HTML saved: {output_file}")
            return str(output_file)

        except Exception as e:
            print(f"❌ HTML generation failed: {e}")
            return None

    def _generate_pdf(self, report: GeneratedReport, base_filename: str) -> str:
        """Generate PDF format using WeasyPrint."""

        if not WEASYPRINT_AVAILABLE:
            print("❌ WeasyPrint not available for PDF generation")
            return None

        try:
            # First generate HTML
            exporter = HTMLExporter()
            html_content = exporter.export(report)

            # Enhanced CSS for better PDF rendering
            enhanced_html = html_content.replace(
                '<style>',
                '''<style>
                @page {
                    size: A4;
                    margin: 1in;
                }
                body {
                    font-family: 'Times New Roman', serif;
                    margin: 0;
                    line-height: 1.6;
                    color: #333;
                }
                h1 {
                    color: #2C3E50;
                    border-bottom: 3px solid #3498DB;
                    page-break-after: avoid;
                }
                h2 {
                    color: #34495E;
                    border-bottom: 1px solid #BDC3C7;
                    page-break-after: avoid;
                    margin-top: 2em;
                }
                h3 {
                    color: #2C3E50;
                    page-break-after: avoid;
                }
                .metadata {
                    background: #F8F9FA;
                    padding: 20px;
                    margin: 20px 0;
                    border-left: 4px solid #3498DB;
                }
                .section {
                    margin: 30px 0;
                    page-break-inside: avoid;
                }
                pre {
                    background: #F8F8F8;
                    padding: 15px;
                    overflow-x: auto;
                    border-left: 4px solid #E74C3C;
                }
                strong {
                    color: #2C3E50;
                }
                a {
                    color: #3498DB;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
                '''
            )

            # Convert to PDF
            output_file = self.output_dir / f"{base_filename}.pdf"
            weasyprint.HTML(string=enhanced_html).write_pdf(str(output_file))

            print(f"✅ PDF saved: {output_file}")
            return str(output_file)

        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            return None


def main():
    """Command line interface for formatting reports."""

    parser = argparse.ArgumentParser(description="Format research reports to multiple formats")
    parser.add_argument("json_file", help="Path to JSON research report")
    parser.add_argument("--formats", "-f", nargs="+",
                       choices=["markdown", "html", "pdf"],
                       default=["markdown", "html", "pdf"],
                       help="Output formats to generate")
    parser.add_argument("--output-dir", "-o", default="formatted_reports",
                       help="Output directory for formatted reports")

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.json_file).exists():
        print(f"❌ Input file not found: {args.json_file}")
        return 1

    # Create formatter and process
    formatter = ResearchReportFormatter(args.output_dir)

    print("🎨 Research Report Formatter")
    print("=" * 50)

    output_files = formatter.format_json_report(args.json_file, args.formats)

    if output_files:
        print("\n🎉 Report formatting complete!")
        print("📁 Generated files:")
        for format_type, file_path in output_files.items():
            if file_path:
                print(f"   {format_type.upper()}: {file_path}")
    else:
        print("\n💥 Report formatting failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())