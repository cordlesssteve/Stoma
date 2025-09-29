#!/usr/bin/env python3
"""
Test script for OpenDeepResearch integration with KnowHunt.

This script tests the basic functionality of the deep research bridge
without requiring external API calls.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from knowhunt.integrations.deep_research_bridge import (
    DeepResearchBridge,
    DeepResearchConfig,
    create_default_deep_research_config,
    OPEN_DEEP_RESEARCH_AVAILABLE
)
from knowhunt.config.deep_research import (
    build_deep_research_config_from_settings,
    is_deep_research_enabled,
    validate_deep_research_config
)
from knowhunt.config.settings import load_config
from knowhunt.pipeline.data_types import NormalizedDocument
from datetime import datetime


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    if OPEN_DEEP_RESEARCH_AVAILABLE:
        print("✅ OpenDeepResearch is available")
    else:
        print("❌ OpenDeepResearch is not available")
        return False

    print("✅ All imports successful")
    return True


def test_configuration():
    """Test configuration loading and validation."""
    print("\n🔧 Testing configuration...")

    # Test default config creation
    default_config = create_default_deep_research_config()
    print(f"✅ Default config created: {default_config.research_model}")

    # Test validation
    validation_errors = validate_deep_research_config(default_config)
    if validation_errors:
        print(f"❌ Validation errors: {validation_errors}")
        return False
    else:
        print("✅ Configuration validation passed")

    # Test loading from settings
    try:
        settings = load_config()
        if is_deep_research_enabled(settings):
            print("✅ Deep research is enabled in settings")
        else:
            print("⚠️  Deep research is disabled in settings")

        settings_config = build_deep_research_config_from_settings(settings)
        print(f"✅ Settings config loaded: {settings_config.research_model}")

    except Exception as e:
        print(f"❌ Settings config error: {e}")
        return False

    return True


def test_bridge_creation():
    """Test creating the bridge without external calls."""
    print("\n🌉 Testing bridge creation...")

    try:
        config = create_default_deep_research_config()
        bridge = DeepResearchBridge(config=config)
        print("✅ Bridge created successfully")

        # Test usage stats
        stats = bridge.get_usage_statistics()
        print(f"✅ Usage stats: {stats['total_analyses']} total analyses")

        return True

    except Exception as e:
        print(f"❌ Bridge creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mock_analysis():
    """Test analysis with a mock document (no external API calls)."""
    print("\n📊 Testing mock analysis...")

    try:
        # Create a test document
        mock_doc = NormalizedDocument(
            id="test_doc_1",
            title="Test Paper: Machine Learning in Healthcare",
            content="This is a test paper about machine learning applications in healthcare. It covers supervised learning, neural networks, and clinical decision support systems.",
            authors=["Test Author"],
            published_date=datetime.now(),
            url="https://example.com/test_paper",
            categories=["cs.LG", "cs.AI"],
            metadata={"test": True}
        )

        print(f"✅ Mock document created: {mock_doc.title}")

        # Note: We won't actually run the analysis as it requires API keys
        # and external services. This is just testing the structure.

        config = create_default_deep_research_config()
        bridge = DeepResearchBridge(config=config)

        print("✅ Bridge ready for analysis")
        print("ℹ️  Skipping actual analysis to avoid API calls")

        return True

    except Exception as e:
        print(f"❌ Mock analysis setup error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_integration():
    """Test CLI command structure."""
    print("\n💻 Testing CLI integration...")

    try:
        # Test that CLI commands can be imported
        from knowhunt.cli.main import deep_research
        print("✅ Deep research CLI group imported successfully")

        # Test that the main CLI still works
        from knowhunt.cli.main import main
        print("✅ Main CLI function imported successfully")

        return True

    except Exception as e:
        print(f"❌ CLI integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting OpenDeepResearch Integration Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_configuration,
        test_bridge_creation,
        test_cli_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print("\n💥 Test failed, stopping here")
            break

    # Run async test
    if passed == total:
        try:
            result = asyncio.run(test_mock_analysis())
            if result:
                passed += 1
            total += 1
        except Exception as e:
            print(f"❌ Async test error: {e}")

    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Integration is ready.")
        print("\nNext steps:")
        print("1. Set up API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, TAVILY_API_KEY)")
        print("2. Test with real analysis: knowhunt deep-research test-integration")
        print("3. Try analyzing papers: knowhunt deep-research analyze-papers -q 'your query'")
    else:
        print("❌ Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)