#!/usr/bin/env python3
"""
Demo script to show morning report generation with existing data
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from knowhunt.reports.morning_pdf_generator import MorningReportPDFGenerator


def main():
    """Generate a demo morning report."""
    print("🌅 Creating Demo Morning Report")
    print("=" * 40)

    # Use the latest existing report
    json_report = "reports/minimal_pipeline/analysis_AI_agents_20250925_174251.json"

    if not Path(json_report).exists():
        print(f"❌ Report file not found: {json_report}")
        return

    # Generate beautiful PDF
    generator = MorningReportPDFGenerator()
    pdf_path = generator.generate_report(json_report, "demo_morning_report.pdf")

    print(f"✅ Demo morning report created: {pdf_path}")
    print("\n🎨 Features included:")
    print("   • Beautiful typography with Inter & Crimson Text fonts")
    print("   • Modern color scheme with gradients")
    print("   • Research quality score visualization")
    print("   • Organized sections with visual hierarchy")
    print("   • Paper summaries with clean formatting")
    print("   • Professional layout perfect for morning reading")
    print("\n☕ Perfect for your morning coffee routine!")


if __name__ == "__main__":
    main()