#!/usr/bin/env python3
"""Remediation verification tests."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_critical_instantiation():
    """Test instantiation of critical components."""
    print("🔍 REMEDIATION VERIFICATION\n")
    
    failures = []
    
    # Test 1: NLP Analyzer instantiation
    print("1. Testing NLP Analyzer instantiation...")
    try:
        from stoma.analysis.nlp_analyzer import NLPAnalyzer
        analyzer = NLPAnalyzer()
        print("   ✅ NLPAnalyzer instantiated successfully")
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        failures.append("NLPAnalyzer instantiation")
    
    # Test 2: Trend Detector instantiation  
    print("2. Testing Trend Detector instantiation...")
    try:
        from stoma.analysis.trend_detector import TrendDetector
        detector = TrendDetector()
        print("   ✅ TrendDetector instantiated successfully")
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        failures.append("TrendDetector instantiation")
    
    # Test 3: Correlation Analyzer instantiation
    print("3. Testing Correlation Analyzer instantiation...")
    try:
        from stoma.analysis.correlation_analyzer import CorrelationAnalyzer
        correlator = CorrelationAnalyzer()
        print("   ✅ CorrelationAnalyzer instantiated successfully")
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        failures.append("CorrelationAnalyzer instantiation")
    
    # Test 4: Batch Processor instantiation
    print("4. Testing Batch Processor instantiation...")
    try:
        from stoma.analysis.batch_processor import BatchProcessor
        processor = BatchProcessor()
        print("   ✅ BatchProcessor instantiated successfully")
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        failures.append("BatchProcessor instantiation")
    
    # Test 5: Report Generator instantiation  
    print("5. Testing Report Generator instantiation...")
    try:
        from stoma.reports.base_generator import DataDrivenReportGenerator
        generator = DataDrivenReportGenerator()
        print("   ✅ DataDrivenReportGenerator instantiated successfully")
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        failures.append("DataDrivenReportGenerator instantiation")
    
    return failures

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\n🧪 BASIC FUNCTIONALITY TESTS\n")
    
    failures = []
    
    # Test NLP Analysis with sample text
    print("1. Testing NLP Analysis functionality...")
    try:
        from stoma.analysis.nlp_analyzer import NLPAnalyzer
        analyzer = NLPAnalyzer()
        
        sample_text = "This is a test document about machine learning and artificial intelligence."
        result = analyzer.analyze(sample_text, "test_doc")
        
        if result.word_count > 0 and result.summary:
            print("   ✅ NLP analysis produces valid results")
        else:
            print("   ❌ FAIL: NLP analysis produces empty results")
            failures.append("NLP analysis functionality")
            
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        failures.append("NLP analysis functionality")
    
    # Test Batch Task Creation
    print("2. Testing Batch Task creation...")
    try:
        from stoma.analysis.batch_processor import BatchProcessor, BatchTask
        from datetime import datetime
        
        processor = BatchProcessor()
        task_id = processor.schedule_nlp_analysis_batch(None, 5, 10)
        
        if task_id and isinstance(task_id, str):
            print("   ✅ Batch task creation works")
        else:
            print("   ❌ FAIL: Batch task creation failed")
            failures.append("Batch task creation")
            
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        failures.append("Batch task creation")
    
    return failures

def test_import_completeness():
    """Test that all expected imports work."""
    print("\n📦 IMPORT COMPLETENESS TESTS\n")
    
    failures = []
    
    # Test analysis module imports
    print("1. Testing analysis module imports...")
    try:
        from stoma.analysis import (
            NLPAnalyzer, NLPService, TrendDetector, 
            CorrelationAnalyzer, BatchProcessor
        )
        print("   ✅ All analysis imports successful")
    except ImportError as e:
        print(f"   ❌ FAIL: Missing analysis import: {e}")
        failures.append("Analysis module imports")
    
    # Test reports module imports
    print("2. Testing reports module imports...")
    try:
        from stoma.reports import ReportGenerator, ReportTemplate
        print("   ✅ Reports imports successful")
    except ImportError as e:
        print(f"   ❌ FAIL: Missing reports import: {e}")
        failures.append("Reports module imports")
    
    return failures

def check_incomplete_implementations():
    """Check for incomplete/stub implementations."""
    print("\n🚧 INCOMPLETE IMPLEMENTATION CHECK\n")
    
    issues = []
    
    # Check for TODO/FIXME comments
    import os
    import re
    
    pattern = re.compile(r'(TODO|FIXME|XXX|HACK|STUB)', re.IGNORECASE)
    
    for root, dirs, files in os.walk('stoma'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern.search(line):
                                issues.append(f"{filepath}:{line_num}: {line.strip()}")
                except:
                    pass
    
    if issues:
        print("   ❌ Found incomplete implementation markers:")
        for issue in issues[:5]:  # Show first 5
            print(f"      {issue}")
        if len(issues) > 5:
            print(f"      ... and {len(issues) - 5} more")
    else:
        print("   ✅ No TODO/FIXME markers found")
    
    return issues

def main():
    """Run all remediation verification tests."""
    print("=" * 60)
    print("🔍 STOMA REMEDIATION VERIFICATION")
    print("=" * 60)
    
    all_failures = []
    
    # Run all tests
    all_failures.extend(test_critical_instantiation())
    all_failures.extend(test_basic_functionality())
    all_failures.extend(test_import_completeness())
    incomplete_issues = check_incomplete_implementations()
    
    print("\n" + "=" * 60)
    print("📊 REMEDIATION RESULTS")
    print("=" * 60)
    
    if not all_failures and not incomplete_issues:
        print("🎉 ALL VERIFICATION GATES PASSED!")
        print("✅ Code compiles successfully")
        print("✅ Components instantiate correctly")  
        print("✅ Basic functionality works")
        print("✅ All imports resolve")
        print("✅ No incomplete implementations detected")
        return 0
    else:
        print("⚠️  VERIFICATION ISSUES FOUND:")
        
        if all_failures:
            print(f"\n❌ Critical Failures ({len(all_failures)}):")
            for failure in all_failures:
                print(f"   - {failure}")
        
        if incomplete_issues:
            print(f"\n🚧 Incomplete Implementations ({len(incomplete_issues)}):")
            print("   - Found TODO/FIXME markers in code")
        
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        print("   1. Fix critical failures before claiming completion")
        print("   2. Complete any stub implementations")
        print("   3. Re-run verification after fixes")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())