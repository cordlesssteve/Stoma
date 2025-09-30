#!/usr/bin/env python3
"""
Beautiful PDF Report Generator for Morning Reading
Creates aesthetic, magazine-style research intelligence reports.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import io
import base64
from typing import Dict, List, Any
import textwrap
import sys
import re

# Import insight extractor
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from enhanced_insight_extractor import InsightExtractor
    INSIGHT_EXTRACTOR_AVAILABLE = True
except ImportError:
    INSIGHT_EXTRACTOR_AVAILABLE = False

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class MorningReportPDFGenerator:
    """Generate beautiful PDF reports for morning reading."""

    def __init__(self, output_dir: str = "reports/morning_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Aesthetic color palette - modern, readable
        self.colors = {
            'primary': '#2C3E50',      # Dark blue-grey
            'secondary': '#3498DB',    # Bright blue
            'accent': '#E74C3C',       # Red accent
            'success': '#27AE60',      # Green
            'warning': '#F39C12',      # Orange
            'text': '#2C3E50',         # Dark text
            'muted': '#7F8C8D',        # Muted grey
            'background': '#ECF0F1',   # Light background
            'white': '#FFFFFF'
        }

        # Typography settings
        self.fonts = {
            'title': ('Arial', 24, 'bold'),
            'heading': ('Arial', 18, 'bold'),
            'subheading': ('Arial', 14, 'bold'),
            'body': ('Arial', 11, 'normal'),
            'caption': ('Arial', 9, 'italic')
        }

    def generate_report(self, json_report_path: str, output_filename: str = None) -> str:
        """Generate a beautiful PDF report from JSON analysis."""

        # Load the JSON report
        with open(json_report_path, 'r') as f:
            report_data = json.load(f)

        # Enhance analysis with better insights if needed
        if INSIGHT_EXTRACTOR_AVAILABLE:
            extractor = InsightExtractor()
            enhanced_analysis = extractor.enhance_analysis(
                report_data.get('analysis', {}),
                report_data.get('papers', [])
            )
            report_data['analysis'] = enhanced_analysis

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            query_safe = report_data.get('query', 'research').replace(' ', '_').replace('/', '_')
            output_filename = f"morning_report_{query_safe}_{timestamp}.pdf"

        output_path = self.output_dir / output_filename

        # Generate the beautiful report
        if WEASYPRINT_AVAILABLE:
            return self._generate_with_weasyprint(report_data, output_path)
        elif REPORTLAB_AVAILABLE:
            return self._generate_with_reportlab(report_data, output_path)
        else:
            raise ImportError("Neither WeasyPrint nor ReportLab available for PDF generation")

    def _generate_with_weasyprint(self, report_data: Dict, output_path: Path) -> str:
        """Generate PDF using WeasyPrint with HTML/CSS for maximum beauty."""

        html_content = self._create_beautiful_html(report_data)

        # Generate PDF
        weasyprint.HTML(string=html_content).write_pdf(str(output_path))
        return str(output_path)

    def _create_beautiful_html(self, report_data: Dict) -> str:
        """Create beautiful HTML content for the report."""

        analysis = report_data.get('analysis', {})
        papers = report_data.get('papers', [])
        timestamp = report_data.get('timestamp', datetime.now().isoformat())

        # Parse timestamp for beautiful display
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_date = dt.strftime("%A, %B %d, %Y")
            formatted_time = dt.strftime("%I:%M %p")
        except:
            formatted_date = "Today"
            formatted_time = "Morning"

        # Generate quality score visualization
        quality_score = analysis.get('research_quality_score', 0)
        quality_bar = self._generate_quality_bar_svg(quality_score)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Morning Research Intelligence Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: {self.colors['text']};
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}

        .container {{
            max-width: 210mm;
            margin: 0 auto;
            background: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            min-height: 297mm;
        }}

        .header {{
            background: linear-gradient(135deg, {self.colors['primary']} 0%, {self.colors['secondary']} 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            transform: rotate(45deg);
        }}

        .header h1 {{
            font-family: 'Crimson Text', serif;
            font-size: 2.5em;
            font-weight: 600;
            margin: 0 0 10px 0;
            position: relative;
            z-index: 1;
        }}

        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
            font-weight: 300;
            position: relative;
            z-index: 1;
        }}

        .header .date {{
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 0.9em;
            opacity: 0.8;
            z-index: 1;
        }}

        .summary-section {{
            padding: 30px 40px;
            background: linear-gradient(to right, #fafafa, white);
            border-left: 4px solid {self.colors['secondary']};
        }}

        .quality-score {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin: 20px 0;
        }}

        .quality-label {{
            font-size: 1.1em;
            font-weight: 600;
            color: {self.colors['primary']};
        }}

        .content {{
            padding: 20px 40px;
        }}

        .section {{
            margin-bottom: 35px;
        }}

        .section h2 {{
            font-family: 'Crimson Text', serif;
            font-size: 1.8em;
            color: {self.colors['primary']};
            border-bottom: 2px solid {self.colors['secondary']};
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .section h3 {{
            font-size: 1.3em;
            color: {self.colors['secondary']};
            margin: 25px 0 15px 0;
            font-weight: 600;
        }}

        .highlights {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .highlight-card {{
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid;
            box-shadow: 0 2px 15px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }}

        .highlight-card.novel {{ border-left-color: {self.colors['success']}; }}
        .highlight-card.technical {{ border-left-color: {self.colors['secondary']}; }}
        .highlight-card.business {{ border-left-color: {self.colors['warning']}; }}

        .highlight-card h4 {{
            margin: 0 0 10px 0;
            font-size: 1.1em;
            font-weight: 600;
        }}

        .highlight-card ul {{
            margin: 0;
            padding-left: 20px;
        }}

        .highlight-card li {{
            margin-bottom: 8px;
            font-size: 0.95em;
            line-height: 1.5;
        }}

        .papers-section {{
            background: #fafafa;
            padding: 30px 40px;
            margin-top: 30px;
        }}

        .paper {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.05);
            border-left: 4px solid {self.colors['accent']};
        }}

        .paper h4 {{
            font-family: 'Crimson Text', serif;
            font-size: 1.4em;
            color: {self.colors['primary']};
            margin: 0 0 10px 0;
            line-height: 1.3;
        }}

        .paper .authors {{
            font-size: 0.9em;
            color: {self.colors['muted']};
            margin-bottom: 15px;
            font-style: italic;
        }}

        .paper .abstract {{
            text-align: justify;
            line-height: 1.6;
            margin-bottom: 10px;
        }}

        .paper .meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            font-size: 0.85em;
            color: {self.colors['muted']};
        }}

        .footer {{
            background: {self.colors['primary']};
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
            opacity: 0.8;
        }}

        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .stat-number {{
            font-size: 2em;
            font-weight: 700;
            color: {self.colors['secondary']};
            display: block;
        }}

        .stat-label {{
            font-size: 0.9em;
            color: {self.colors['muted']};
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="date">{formatted_date}<br>{formatted_time}</div>
            <h1>Morning Research Intelligence</h1>
            <div class="subtitle">Your Daily AI & Technology Briefing</div>
        </div>

        <div class="summary-section">
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-number">{len(papers)}</span>
                    <div class="stat-label">Papers Analyzed</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{quality_score}/10</span>
                    <div class="stat-label">Quality Score</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{len(analysis.get('novel_contributions', []))}</span>
                    <div class="stat-label">Novel Insights</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{report_data.get('model', 'N/A').split(':')[0].upper()}</span>
                    <div class="stat-label">AI Model</div>
                </div>
            </div>
        </div>

        <div class="content">
            <div class="section">
                <h2>Key Intelligence Summary</h2>
                <div class="highlights">
                    <div class="highlight-card novel">
                        <h4>🔬 Novel Contributions</h4>
                        <ul>
        """

        # Add novel contributions (with fallback enhancement)
        contributions = analysis.get('novel_contributions', [])
        for contribution in contributions[:5]:
            # If it's just a paper title, try to extract meaningful insight
            if contribution in [p.get('title', '') for p in papers]:
                # Find the paper and extract a key insight from its abstract
                matching_paper = next((p for p in papers if p.get('title') == contribution), None)
                if matching_paper:
                    abstract = matching_paper.get('abstract', '')
                    # Extract first sentence that contains technical terms
                    sentences = abstract.split('. ')
                    insight = next((s for s in sentences if any(term in s.lower() for term in
                                   ['method', 'approach', 'algorithm', 'technique', 'model', 'system', 'framework'])),
                                 contribution)
                    html += f"<li><strong>From {contribution.split(':')[0]}:</strong> {insight[:150]}...</li>"
                else:
                    html += f"<li>{contribution}</li>"
            else:
                html += f"<li>{contribution}</li>"

        html += """
                        </ul>
                    </div>
                    <div class="highlight-card technical">
                        <h4>⚡ Technical Innovations</h4>
                        <ul>
        """

        # Add technical innovations (with fallback enhancement)
        innovations = analysis.get('technical_innovations', [])
        for innovation in innovations[:5]:
            # If it's just a paper title, extract technical insight
            if innovation in [p.get('title', '') for p in papers]:
                matching_paper = next((p for p in papers if p.get('title') == innovation), None)
                if matching_paper:
                    abstract = matching_paper.get('abstract', '')
                    sentences = abstract.split('. ')
                    tech_insight = next((s for s in sentences if any(term in s.lower() for term in
                                        ['performance', 'accuracy', 'efficiency', 'speed', 'novel', 'new', 'improvement'])),
                                      innovation)
                    html += f"<li><strong>Technical:</strong> {tech_insight[:150]}...</li>"
                else:
                    html += f"<li>{innovation}</li>"
            else:
                html += f"<li>{innovation}</li>"

        html += """
                        </ul>
                    </div>
                    <div class="highlight-card business">
                        <h4>💼 Business Implications</h4>
                        <ul>
        """

        # Add business implications (with fallback enhancement)
        implications = analysis.get('business_implications', [])
        for implication in implications[:5]:
            # If it's just a paper title, extract business insight
            if implication in [p.get('title', '') for p in papers]:
                matching_paper = next((p for p in papers if p.get('title') == implication), None)
                if matching_paper:
                    abstract = matching_paper.get('abstract', '')
                    sentences = abstract.split('. ')
                    business_insight = next((s for s in sentences if any(term in s.lower() for term in
                                           ['cost', 'market', 'business', 'commercial', 'application', 'industry', 'deployment'])),
                                          implication)
                    html += f"<li><strong>Business:</strong> {business_insight[:150]}...</li>"
                else:
                    html += f"<li>{implication}</li>"
            else:
                html += f"<li>{implication}</li>"

        html += f"""
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <div class="papers-section">
            <h2 style="color: {self.colors['primary']}; text-align: center; margin-bottom: 30px; font-family: 'Crimson Text', serif; font-size: 1.8em;">Featured Research Papers</h2>
        """

        # Add papers
        for i, paper in enumerate(papers[:5]):
            # Truncate abstract for readability
            abstract = paper.get('abstract', '')
            if len(abstract) > 400:
                abstract = abstract[:400] + "..."

            # Format authors nicely
            authors = paper.get('authors', [])
            if len(authors) > 3:
                authors_str = f"{', '.join(authors[:3])} et al. ({len(authors)} authors)"
            else:
                authors_str = ', '.join(authors)

            # Parse date
            published = paper.get('published', '')
            try:
                pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                formatted_pub = pub_date.strftime("%B %d, %Y")
            except:
                formatted_pub = published

            html += f"""
            <div class="paper">
                <h4>{paper.get('title', 'Untitled Paper')}</h4>
                <div class="authors">{authors_str}</div>
                <div class="abstract">{abstract}</div>
                <div class="meta">
                    <span>Published: {formatted_pub}</span>
                    <span>ArXiv ID: {paper.get('arxiv_id', 'N/A')}</span>
                </div>
            </div>
            """

        html += f"""
        </div>

        <div class="footer">
            Generated by KnowHunt Research Intelligence System<br>
            Report created on {formatted_date} • {len(papers)} papers analyzed • Quality score: {quality_score}/10
        </div>
    </div>
</body>
</html>
        """

        return html

    def _generate_quality_bar_svg(self, score: int) -> str:
        """Generate SVG for quality score visualization."""
        width = 200
        height = 20
        fill_width = (score / 10) * width

        color = self.colors['success'] if score >= 8 else self.colors['warning'] if score >= 6 else self.colors['accent']

        return f'''
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="{width}" height="{height}" fill="{self.colors['background']}" rx="10"/>
            <rect width="{fill_width}" height="{height}" fill="{color}" rx="10"/>
        </svg>
        '''


def main():
    """Test the PDF generator."""
    generator = MorningReportPDFGenerator()

    # Find the most recent report
    reports_dir = Path("reports/minimal_pipeline")
    if reports_dir.exists():
        json_files = list(reports_dir.glob("*.json"))
        if json_files:
            latest_report = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"Generating PDF from: {latest_report}")

            output_path = generator.generate_report(str(latest_report))
            print(f"✅ Beautiful PDF report generated: {output_path}")
            return output_path
        else:
            print("❌ No JSON reports found in reports/minimal_pipeline")
    else:
        print("❌ Reports directory not found")

    return None


if __name__ == "__main__":
    main()