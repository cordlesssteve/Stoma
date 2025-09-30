"""Export formats for generated reports."""

from abc import ABC, abstractmethod
from typing import Any
from .base_generator import GeneratedReport, ReportSection


class ReportExporter(ABC):
    """Abstract base class for report exporters."""
    
    @abstractmethod
    def export(self, report: GeneratedReport) -> str:
        """Export report to string format."""
        pass


class MarkdownExporter(ReportExporter):
    """Export reports to Markdown format."""
    
    def export(self, report: GeneratedReport) -> str:
        """Export report to Markdown."""
        lines = []
        
        # Title
        lines.append(f"# {report.title}")
        lines.append("")
        
        # Metadata
        lines.append("---")
        lines.append(f"**Report ID:** {report.report_id}")
        lines.append(f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Template:** {report.template_id}")
        lines.append("---")
        lines.append("")
        
        # Sections
        for section in report.sections:
            lines.append(self._format_section(section))
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_section(self, section: ReportSection) -> str:
        """Format a section in Markdown."""
        lines = []
        
        # Section title (already has ## from content usually)
        if not section.content.startswith('#'):
            lines.append(f"## {section.title}")
            lines.append("")
        
        lines.append(section.content)
        
        # Subsections
        for subsection in section.subsections:
            lines.append("")
            lines.append(self._format_section(subsection))
        
        return "\n".join(lines)


class HTMLExporter(ReportExporter):
    """Export reports to HTML format."""
    
    def export(self, report: GeneratedReport) -> str:
        """Export report to HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        .metadata {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
        .section {{ margin: 20px 0; }}
        pre {{ background: #f8f8f8; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    
    <div class="metadata">
        <strong>Report ID:</strong> {report.report_id}<br>
        <strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>Template:</strong> {report.template_id}
    </div>
"""
        
        # Sections
        for section in report.sections:
            html += f'<div class="section">\n'
            html += self._markdown_to_html(section.content)
            html += '</div>\n'
        
        html += """
</body>
</html>"""
        
        return html
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """Convert basic Markdown to HTML."""
        lines = markdown_content.split('\n')
        html_lines = []
        
        for line in lines:
            if line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('**') and line.endswith('**'):
                html_lines.append(f'<strong>{line[2:-2]}</strong>')
            elif line.startswith('- '):
                html_lines.append(f'<li>{line[2:]}</li>')
            elif line.strip() == '':
                html_lines.append('<br>')
            else:
                html_lines.append(f'<p>{line}</p>')
        
        return '\n'.join(html_lines)


class PDFExporter(ReportExporter):
    """Export reports to PDF format."""
    
    def export(self, report: GeneratedReport) -> str:
        """Export report to PDF (returns HTML for now)."""
        # For now, return HTML that can be converted to PDF
        # In a full implementation, this would use libraries like reportlab or weasyprint
        html_exporter = HTMLExporter()
        return html_exporter.export(report)