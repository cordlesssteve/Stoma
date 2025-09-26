#!/usr/bin/env python3
"""
Morning Report Generator - CLI for beautiful PDF reports
Usage: python3 generate_morning_report.py [query] [--model MODEL]
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

# Add the project root to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from minimal_pipeline import run_minimal_pipeline
from knowhunt.reports.morning_pdf_generator import MorningReportPDFGenerator


async def generate_morning_report(query: str = "AI agents", count: int = 5, model: str = "codellama:13b-instruct"):
    """Generate a complete morning report with analysis and beautiful PDF."""

    print(f"🌅 Generating Morning Research Report")
    print(f"📋 Query: {query}")
    print(f"📊 Papers: {count}")
    print(f"🤖 Model: {model}")
    print("=" * 60)

    # Step 1: Run the analysis pipeline
    print("📡 Collecting and analyzing papers...")
    try:
        json_report_path = await run_minimal_pipeline(query, count, model)
        print(f"✅ Analysis complete: {json_report_path}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

    # Step 2: Generate beautiful PDF
    print("🎨 Creating beautiful PDF report...")
    try:
        generator = MorningReportPDFGenerator()
        pdf_path = generator.generate_report(json_report_path)
        print(f"✅ PDF report generated: {pdf_path}")

        # Show summary
        with open(json_report_path, 'r') as f:
            data = json.load(f)
            analysis = data.get('analysis', {})
            quality_score = analysis.get('research_quality_score', 0)
            novel_count = len(analysis.get('novel_contributions', []))

        print(f"\n📊 Report Summary:")
        print(f"   Quality Score: {quality_score}/10")
        print(f"   Novel Insights: {novel_count}")
        print(f"   Papers Analyzed: {len(data.get('papers', []))}")
        print(f"   Model Used: {model}")

        return pdf_path

    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return json_report_path  # Return JSON as fallback


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate beautiful morning research intelligence reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_morning_report.py "machine learning"
  python3 generate_morning_report.py "protein folding" --model "codellama:13b-instruct"
  python3 generate_morning_report.py "quantum computing" --count 10
        """
    )

    parser.add_argument(
        "query",
        nargs='?',
        default="AI agents",
        help="Research query to analyze (default: 'AI agents')"
    )

    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="Number of papers to analyze (default: 5)"
    )

    parser.add_argument(
        "--model", "-m",
        default="codellama:13b-instruct",
        help="LLM model to use (default: 'codellama:13b-instruct')"
    )

    args = parser.parse_args()

    # Run the morning report generation
    try:
        result = asyncio.run(generate_morning_report(args.query, args.count, args.model))
        if result:
            print(f"\n🎉 Your morning report is ready: {result}")
            print("Perfect for reading with your morning coffee! ☕")
        else:
            print("\n❌ Failed to generate morning report")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Report generation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()