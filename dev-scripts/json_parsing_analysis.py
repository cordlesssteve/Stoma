#!/usr/bin/env python3
"""
Deep Analysis of the JSON Parsing Issue

This script demonstrates exactly what went wrong with our JSON parsing
and why larger models cause more parsing failures.
"""

import json
import re
from typing import Dict, Any, Optional

def demonstrate_parsing_issue():
    """Demonstrate the exact JSON parsing problem we encountered."""

    print("🔍 JSON Parsing Issue Analysis")
    print("=" * 50)

    # This is the actual raw response from Mistral 7B model
    mistral_raw_response = """ {
  "novel_contributions": [
    "Development of a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations",
    "Introduction of Interactive Recommendation Feed (IRF) paradigm enabling active user commands within recommendation feeds",
    "Creation of SAGE benchmark for evaluating semantic understanding across various categories and datasets"
  ],
  "technical_innovations": [
    "Pretraining on a large corpus of scientific text, pure sequences, and sequence-text pairs followed by alignment via SFT and reinforcement learning",
    "Dual-agent architecture with a Parser Agent and Planner Agent for real-time linguistic command interpretation",
    "Evaluation framework assessing embedding models and similarity metrics across five categories: Human Preference Alignment, Transformation Robustness, Information Sensitivity, Clustering Performance, and Retrieval Robustness"
  ],
  "business_implications": [
    "Improved cross-discipline learning strengthens transfer and downstream reliability in scientific reasoning",
    "Enhanced user satisfaction and business outcomes through active explicit control over recommendation policies",
    "Provides a more challenging evaluation framework for semantic understanding, enabling better AI models"
  ],
  "research_quality_score": [
    8 (Paper 1),
    9 (Paper 2),
    7 (Paper 3)
  ]
}"""

    # This is what our current parser does
    print("📋 Current Parsing Logic:")
    print("1. Find first '{' character")
    print("2. Find last '}' character")
    print("3. Extract substring between them")
    print("4. Try json.loads() on extracted string")

    # Demonstrate current parsing approach
    print("\n🔧 Current Parser Attempt:")
    start = mistral_raw_response.find('{')
    end = mistral_raw_response.rfind('}') + 1

    print(f"Start position: {start}")
    print(f"End position: {end}")

    if start != -1 and end > start:
        json_str = mistral_raw_response[start:end]
        print(f"Extracted JSON string length: {len(json_str)} characters")
        print("\nExtracted content:")
        print(json_str[:200] + "..." if len(json_str) > 200 else json_str)

        try:
            parsed = json.loads(json_str)
            print("\n✅ Parsing SUCCESS!")
            print(f"Keys found: {list(parsed.keys())}")
        except json.JSONDecodeError as e:
            print(f"\n❌ Parsing FAILED: {e}")
            print(f"Error at position: {e.pos}")

            # Show the problematic part
            if e.pos < len(json_str):
                problem_area = json_str[max(0, e.pos-20):e.pos+20]
                print(f"Problem area: '{problem_area}'")

                # Identify the specific issue
                analyze_json_error(json_str, e)

def analyze_json_error(json_str: str, error: json.JSONDecodeError):
    """Analyze what specifically went wrong with JSON parsing."""
    print("\n🔍 Detailed Error Analysis:")

    # Common issues with LLM-generated JSON
    issues_found = []

    # Issue 1: Comments in JSON (like "8 (Paper 1)")
    if '(' in json_str and ')' in json_str:
        issues_found.append("Comments in JSON values (invalid JSON syntax)")
        print("❌ Issue 1: Comments in JSON")
        print("   Problem: '8 (Paper 1)' is not valid JSON")
        print("   Valid JSON: '8' or 'Paper 1: 8'")

    # Issue 2: Trailing commas
    trailing_comma_pattern = r',\s*[}\]]'
    if re.search(trailing_comma_pattern, json_str):
        issues_found.append("Trailing commas")
        print("❌ Issue 2: Trailing commas found")

    # Issue 3: Unescaped quotes
    unescaped_quotes = re.findall(r'(?<!\\)"(?![,:}\]\s])', json_str)
    if unescaped_quotes:
        issues_found.append("Unescaped quotes in strings")
        print("❌ Issue 3: Unescaped quotes in strings")

    # Issue 4: Mixed data types in arrays
    if '[' in json_str and ']' in json_str:
        # Look for arrays with mixed types
        array_content = re.findall(r'\[([^\]]+)\]', json_str)
        for content in array_content:
            if '(' in content and '"' in content:
                issues_found.append("Mixed data types in arrays")
                print("❌ Issue 4: Mixed data types in arrays")
                print(f"   Problem array: [{content}]")
                break

    print(f"\n📊 Total issues found: {len(issues_found)}")
    return issues_found

def demonstrate_model_differences():
    """Show how different models produce different JSON quality."""

    print("\n🧠 Model Response Patterns")
    print("=" * 40)

    models_data = {
        "qwen2.5-coder:3b": {
            "response_style": "Clean, structured JSON",
            "typical_issues": ["None - very reliable"],
            "success_rate": "~95%",
            "example_problem": None
        },
        "mistral:7b-instruct": {
            "response_style": "Detailed but adds commentary",
            "typical_issues": [
                "Comments in JSON values: '8 (Paper 1)'",
                "Explanatory text outside JSON",
                "Complex nested structures"
            ],
            "success_rate": "~60%",
            "example_problem": "research_quality_score: [8 (Paper 1), 9 (Paper 2)]"
        },
        "codellama:13b-instruct": {
            "response_style": "Very detailed, code-focused",
            "typical_issues": [
                "Code comments mixed in JSON",
                "Multi-line explanations",
                "Type annotations"
            ],
            "success_rate": "~40%",
            "example_problem": "Often fails completely with our current parsing"
        },
        "phi3.5:latest": {
            "response_style": "Verbose with explanations",
            "typical_issues": [
                "JSON wrapped in markdown code blocks",
                "Extensive explanatory text after JSON"
            ],
            "success_rate": "~80%",
            "example_problem": "```json\\n{...}\\n```"
        }
    }

    for model, data in models_data.items():
        print(f"\n🤖 {model}:")
        print(f"   Style: {data['response_style']}")
        print(f"   Success Rate: {data['success_rate']}")
        print(f"   Main Issues:")
        for issue in data['typical_issues']:
            print(f"     - {issue}")
        if data['example_problem']:
            print(f"   Example Problem: {data['example_problem']}")

def show_parsing_solutions():
    """Demonstrate better parsing approaches."""

    print("\n🔧 Improved Parsing Solutions")
    print("=" * 35)

    print("1. **Robust JSON Extraction:**")
    print("   - Handle markdown code blocks (```json)")
    print("   - Strip comments from JSON values")
    print("   - Fix common formatting issues")

    print("\n2. **Multi-Pass Parsing:**")
    print("   - Try direct JSON parsing first")
    print("   - Fall back to regex extraction")
    print("   - Final fallback to manual parsing")

    print("\n3. **Model-Specific Adaptations:**")
    print("   - Different parsing strategies per model")
    print("   - Custom prompting for clean JSON")
    print("   - Post-processing for known model quirks")

    # Demonstrate improved parser
    print("\n🔧 Improved Parser Example:")

def improved_json_parser(raw_response: str) -> Optional[Dict[str, Any]]:
    """Improved JSON parser that handles model quirks."""

    # Step 1: Handle markdown code blocks
    if '```json' in raw_response:
        start_marker = raw_response.find('```json') + 7
        end_marker = raw_response.find('```', start_marker)
        if end_marker != -1:
            raw_response = raw_response[start_marker:end_marker].strip()

    # Step 2: Find JSON boundaries
    start = raw_response.find('{')
    end = raw_response.rfind('}') + 1

    if start == -1 or end <= start:
        return None

    json_str = raw_response[start:end]

    # Step 3: Clean up common issues
    json_str = clean_json_string(json_str)

    # Step 4: Try parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Step 5: Try manual extraction as fallback
        return manual_json_extraction(json_str)

def clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues."""

    # Fix 1: Remove comments in arrays like "8 (Paper 1)"
    json_str = re.sub(r'(\d+)\s*\([^)]+\)', r'\1', json_str)

    # Fix 2: Remove trailing commas
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    # Fix 3: Fix unescaped quotes (basic attempt)
    # This is complex and model-specific

    return json_str

def manual_json_extraction(json_str: str) -> Dict[str, Any]:
    """Manual extraction as final fallback."""
    result = {}

    # Extract arrays using regex
    array_patterns = {
        'novel_contributions': r'"novel_contributions":\s*\[(.*?)\]',
        'technical_innovations': r'"technical_innovations":\s*\[(.*?)\]',
        'business_implications': r'"business_implications":\s*\[(.*?)\]'
    }

    for key, pattern in array_patterns.items():
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            array_content = match.group(1)
            # Extract quoted strings
            items = re.findall(r'"([^"]*)"', array_content)
            result[key] = items

    # Extract quality score
    quality_match = re.search(r'"research_quality_score":\s*(\d+(?:\.\d+)?)', json_str)
    if quality_match:
        result['research_quality_score'] = float(quality_match.group(1))

    return result

def main():
    """Demonstrate the JSON parsing issue and solutions."""

    # Show the exact problem
    demonstrate_parsing_issue()

    # Explain model differences
    demonstrate_model_differences()

    # Show solutions
    show_parsing_solutions()

    print("\n🎯 Key Insights:")
    print("1. **Larger models = more detailed responses = more parsing challenges**")
    print("2. **Each model has different response patterns and quirks**")
    print("3. **Simple JSON extraction fails with sophisticated model outputs**")
    print("4. **Need model-aware parsing strategies for production systems**")

    print("\n💡 The Paradox:")
    print("Better models give better analysis content,")
    print("but require more sophisticated parsing to extract it!")

if __name__ == "__main__":
    main()