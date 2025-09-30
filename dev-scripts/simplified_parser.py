#!/usr/bin/env python3
"""
Simplified Parser - The Right Way

The LLM responses are already mostly clean JSON. We just need minimal preprocessing.
"""

import json
import re
from typing import Dict, Any, Optional

def simplified_json_parser(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Simplified parser that assumes LLM responses are mostly good JSON.

    The key insight: Don't overthink it - LLMs already return clean data!
    """

    # Step 1: Handle markdown code blocks (if present)
    if '```json' in raw_response:
        start = raw_response.find('```json') + 7
        end = raw_response.find('```', start)
        if end != -1:
            raw_response = raw_response[start:end].strip()

    # Step 2: Extract JSON boundaries
    start = raw_response.find('{')
    end = raw_response.rfind('}') + 1

    if start == -1 or end <= start:
        return None

    json_str = raw_response[start:end].strip()

    # Step 3: Fix the ONE common issue - comments in JSON values
    # "research_quality_score": [8 (Paper 1), 9 (Paper 2)]
    # becomes: "research_quality_score": [8, 9]
    json_str = re.sub(r'(\d+)\s*\([^)]+\)', r'\1', json_str)

    # Step 4: Try to parse - that's it!
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Don't fall back to keyword extraction - just fail with useful error
        print(f"JSON parsing failed at position {e.pos}: {e.msg}")
        print(f"Problematic area: {json_str[max(0, e.pos-20):e.pos+20]}")
        return None

# Test it on our actual responses
def test_simplified_parser():
    """Test the simplified parser on real model responses."""

    # Real response from mistral:7b-instruct that failed with our current system
    test_response = """{
  "novel_contributions": [
    "Development of a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations",
    "Introduction of Interactive Recommendation Feed (IRF) paradigm enabling active user commands within recommendation feeds",
    "Creation of SAGE benchmark for evaluating semantic understanding across various categories and datasets"
  ],
  "technical_innovations": [
    "Pretraining on a large corpus of scientific text, pure sequences, and sequence-text pairs followed by alignment via SFT and reinforcement learning",
    "Dual-agent architecture with a Parser Agent and Planner Agent for real-time linguistic command interpretation"
  ],
  "business_implications": [
    "Improved cross-discipline learning strengthens transfer and downstream reliability in scientific reasoning",
    "Enhanced user satisfaction and business outcomes through active explicit control over recommendation policies"
  ],
  "research_quality_score": [8 (Paper 1), 9 (Paper 2), 7 (Paper 3)]
}"""

    print("🧪 Testing Simplified Parser")
    print("=" * 30)

    result = simplified_json_parser(test_response)

    if result:
        print("✅ Parsing SUCCESS!")
        print(f"Novel contributions: {len(result['novel_contributions'])}")
        print(f"Technical innovations: {len(result['technical_innovations'])}")
        print(f"Business implications: {len(result['business_implications'])}")
        print(f"Research quality score: {result['research_quality_score']}")
        print("\nFirst contribution:")
        print(f"  {result['novel_contributions'][0]}")
    else:
        print("❌ Parsing failed")

if __name__ == "__main__":
    test_simplified_parser()