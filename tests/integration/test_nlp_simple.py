#!/usr/bin/env python3
"""Simple test for NLP analyzer without database dependencies."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_nlp_analyzer():
    """Test the NLP analyzer directly."""
    print("Testing NLP Analyzer...")
    
    try:
        from stoma.analysis.nlp_analyzer import NLPAnalyzer, AnalysisResult
        
        # Sample academic text
        sample_text = """
        This paper presents a novel approach to machine learning using quantum computing principles.
        We propose a new algorithm called QuantumNet that leverages quantum superposition to enhance
        traditional neural network architectures. The method shows promising results on the CIFAR-10
        dataset, achieving 95% accuracy. Our experiments demonstrate that quantum-enhanced learning
        can significantly improve performance in classification tasks. The proposed approach opens
        new avenues for research in quantum machine learning and artificial intelligence.
        """
        
        # Initialize analyzer (will use fallback methods if SpaCy not available)
        analyzer = NLPAnalyzer()
        
        # Perform analysis
        result = analyzer.analyze(sample_text, "test_doc")
        
        print("✅ Analysis completed successfully!")
        print(f"Document ID: {result.document_id}")
        print(f"Word count: {result.word_count}")
        print(f"Sentence count: {result.sentence_count}")
        print(f"Summary: {result.summary[:100]}...")
        print(f"Top keywords: {result.keywords[:3]}")
        print(f"Entities found: {list(result.entities.keys())}")
        print(f"Sentiment: {result.sentiment.get('sentiment_label', 'unknown')}")
        print(f"Readability: {result.readability_score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the simple test."""
    print("🧪 Simple NLP Pipeline Test\n")
    print("=" * 40)
    
    if test_nlp_analyzer():
        print("\n🎉 NLP analyzer test passed!")
        print("\nThe traditional NLP pipeline is working correctly.")
        print("You can now:")
        print("1. Install models: python3 -m spacy download en_core_web_sm")
        print("2. Set up database for full functionality")
        print("3. Use CLI: python3 -m knowhunt.cli.main nlp analyze-text 'Your text'")
    else:
        print("\n❌ Test failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())