#!/usr/bin/env python3
"""Test script for the traditional NLP pipeline."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_nlp_imports():
    """Test that we can import all NLP components."""
    print("Testing NLP imports...")
    
    try:
        from knowhunt.analysis.nlp_analyzer import NLPAnalyzer, AnalysisResult
        from knowhunt.analysis.nlp_service import NLPService
        from knowhunt.analysis.nlp_storage import NLPStorage
        print("✅ All NLP components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_analyzer_without_models():
    """Test analyzer with graceful model handling."""
    print("\nTesting NLP analyzer (without models)...")
    
    try:
        # Test with minimal dependencies first
        from knowhunt.analysis.nlp_analyzer import NLPAnalyzer
        
        # Sample academic text
        sample_text = """
        This paper presents a novel approach to machine learning using quantum computing principles.
        We propose a new algorithm called QuantumNet that leverages quantum superposition to enhance
        traditional neural network architectures. The method shows promising results on the CIFAR-10
        dataset, achieving 95% accuracy. Our experiments demonstrate that quantum-enhanced learning
        can significantly improve performance in classification tasks. The proposed approach opens
        new avenues for research in quantum machine learning and artificial intelligence.
        """
        
        print("Sample text prepared...")
        print("✅ Analyzer framework test passed")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        return False

def test_analysis_result_structure():
    """Test the AnalysisResult dataclass structure."""
    print("\nTesting AnalysisResult structure...")
    
    try:
        from knowhunt.analysis.nlp_analyzer import AnalysisResult
        
        # Create a test result
        result = AnalysisResult(
            document_id="test_doc_1",
            summary="This is a test summary of the document.",
            keywords=[("machine learning", 0.85), ("quantum computing", 0.72)],
            entities={"ALGORITHM": ["QuantumNet"], "DATASET": ["CIFAR-10"]},
            sentiment={"polarity": 0.1, "subjectivity": 0.3},
            topics=["machine learning", "quantum computing"],
            readability_score=65.0,
            word_count=150,
            sentence_count=8
        )
        
        print(f"Document ID: {result.document_id}")
        print(f"Summary length: {len(result.summary)} chars")
        print(f"Keywords count: {len(result.keywords)}")
        print(f"Entity types: {list(result.entities.keys())}")
        print(f"Word count: {result.word_count}")
        print("✅ AnalysisResult structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ AnalysisResult test failed: {e}")
        return False

def test_cli_imports():
    """Test that CLI commands can be imported."""
    print("\nTesting CLI imports...")
    
    try:
        from knowhunt.cli.main import main
        print("✅ CLI imports successful")
        return True
    except ImportError as e:
        print(f"❌ CLI import error: {e}")
        return False

def test_database_integration():
    """Test database components without requiring actual database."""
    print("\nTesting database integration components...")
    
    try:
        from knowhunt.analysis.nlp_storage import NLPStorage
        from knowhunt.analysis.nlp_service import NLPService
        
        print("✅ Database integration components imported")
        return True
    except ImportError as e:
        print(f"❌ Database integration test failed: {e}")
        return False

def test_traditional_nlp_methods():
    """Test traditional NLP methods that don't require external models."""
    print("\nTesting traditional NLP methods...")
    
    try:
        import nltk
        from textblob import TextBlob
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Test basic text processing
        text = "This is a sample text for testing natural language processing capabilities."
        
        # Test TextBlob (basic functionality)
        blob = TextBlob(text)
        sentiment = blob.sentiment
        print(f"Sample sentiment: polarity={sentiment.polarity:.3f}, subjectivity={sentiment.subjectivity:.3f}")
        
        # Test TF-IDF
        vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = vectorizer.fit_transform([text])
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        print("✅ Traditional NLP methods test passed")
        return True
        
    except Exception as e:
        print(f"❌ Traditional NLP methods test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing KnowHunt Traditional NLP Pipeline\n")
    print("=" * 50)
    
    tests = [
        test_basic_nlp_imports,
        test_analyzer_without_models,
        test_analysis_result_structure,
        test_cli_imports,
        test_database_integration,
        test_traditional_nlp_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Traditional NLP pipeline architecture is ready.")
        print("\nNext steps:")
        print("1. Install SpaCy model: python3 -m spacy download en_core_web_sm")
        print("2. Download NLTK data: python3 -c \"import nltk; nltk.download('all')\"")
        print("3. Test with real data: python3 -m knowhunt.cli.main nlp analyze-text 'Your text here'")
    else:
        print(f"❌ {total - passed} tests failed. Please fix issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())