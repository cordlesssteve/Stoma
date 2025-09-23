#!/usr/bin/env python3
"""
Test script for Ollama local LLM integration.

This demonstrates the local model integration for high-powered research analysis
using Llama 3.1, DeepSeek, and other locally-managed models.
"""

import asyncio
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test if we can import the LLM modules
try:
    from knowhunt.analysis.llm_analyzer import LLMAnalyzer, LLMAnalysisService
    LLM_AVAILABLE = True
except ImportError as e:
    logger.error(f"LLM modules not available: {e}")
    LLM_AVAILABLE = False


# Recommended local models from Phase 2 strategy
RECOMMENDED_MODELS = {
    "llama3.1:70b": {
        "description": "Llama-3.1-70B-Instruct for general research analysis",
        "use_case": "Comprehensive research understanding and analysis",
        "memory_req": "~40GB VRAM"
    },
    "llama3.1:8b": {
        "description": "Llama-3.1-8B-Instruct for lightweight analysis",
        "use_case": "Fast analysis for smaller papers or summaries",
        "memory_req": "~8GB VRAM"
    },
    "deepseek-coder:33b": {
        "description": "DeepSeek-Coder-V2 for technical/methods extraction",
        "use_case": "Code analysis, technical method extraction",
        "memory_req": "~20GB VRAM"
    },
    "qwen2.5:72b": {
        "description": "Qwen2.5-72B-Instruct for research writing",
        "use_case": "High-quality research summaries and reports",
        "memory_req": "~40GB VRAM"
    },
    "mistral:7b": {
        "description": "Mistral-7B for fast prototyping",
        "use_case": "Quick testing and development",
        "memory_req": "~6GB VRAM"
    }
}


async def check_ollama_status():
    """Check if Ollama is running and what models are available."""
    print("🔍 Checking Ollama Status")
    print("=" * 40)
    
    if not LLM_AVAILABLE:
        print("❌ LLM modules not available.")
        return False, []
    
    try:
        # Create a test analyzer to check Ollama connectivity
        analyzer = LLMAnalyzer(provider="ollama", model="test")
        
        # Check what models are available
        available = await analyzer._check_ollama_model_availability()
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    result = await response.json()
                    models = [model['name'] for model in result.get('models', [])]
                    
                    print("✅ Ollama is running!")
                    print(f"📦 Available models: {len(models)}")
                    
                    if models:
                        print("\n🎯 Currently Installed Models:")
                        for model in models:
                            print(f"   • {model}")
                    else:
                        print("\n⚠️  No models installed yet.")
                    
                    return True, models
                else:
                    print(f"❌ Ollama API returned status {response.status}")
                    return False, []
                    
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("\n💡 To install Ollama:")
        print("   1. Visit: https://ollama.ai/download")
        print("   2. Install Ollama for your system")
        print("   3. Start Ollama service")
        print("   4. Pull a model: ollama pull llama3.1:8b")
        return False, []


async def test_model_recommendations():
    """Show recommended models and installation commands."""
    print("\n🎯 Recommended Models for Research Analysis")
    print("=" * 50)
    
    print("Based on Phase 2 strategy, here are the optimal models:\n")
    
    for model_name, info in RECOMMENDED_MODELS.items():
        print(f"📋 **{model_name}**")
        print(f"   Description: {info['description']}")
        print(f"   Use Case: {info['use_case']}")
        print(f"   Memory Req: {info['memory_req']}")
        print(f"   Install: ollama pull {model_name}")
        print()
    
    print("💡 **Recommended Setup Strategy:**")
    print("1. Start with llama3.1:8b for testing (lightweight)")
    print("2. Add llama3.1:70b for production analysis (if you have 40GB+ VRAM)")
    print("3. Consider deepseek-coder:33b for technical papers")
    print("4. Use mistral:7b for rapid prototyping")


async def test_local_model_analysis(model_name: str):
    """Test analysis with a specific local model."""
    print(f"\n🧪 Testing Analysis with {model_name}")
    print("=" * 50)
    
    if not LLM_AVAILABLE:
        print("❌ LLM modules not available.")
        return False
    
    # Sample research paper for testing
    test_paper = {
        "title": "Efficient Parameter-Efficient Fine-Tuning with LoRA Adapters",
        "content": """
        Abstract: We propose an efficient approach to parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) adapters. Our method reduces the number of trainable parameters by 90% while maintaining comparable performance to full fine-tuning. The key innovation is dynamic rank selection based on layer importance analysis.

        Introduction: Large language models require substantial computational resources for fine-tuning. Parameter-efficient methods like LoRA offer promising alternatives but typically use fixed rank values across all layers.

        Method: Our approach dynamically selects optimal rank values for each transformer layer based on gradient analysis and layer importance scoring. This allows for more efficient parameter allocation.

        Results: Experiments on GLUE benchmarks show that dynamic LoRA achieves 98.5% of full fine-tuning performance while using only 10% of the parameters.

        Conclusion: Dynamic rank selection significantly improves parameter efficiency in LoRA-based fine-tuning.
        """,
        "document_id": "dynamic_lora_test"
    }
    
    try:
        # Initialize analyzer with local model
        analyzer = LLMAnalyzer(
            provider="ollama", 
            model=model_name,
            max_tokens=1500,
            temperature=0.1
        )
        
        print(f"🚀 Initializing analysis with {model_name}...")
        
        # Test basic analysis
        result = await analyzer.analyze_research_paper(
            text=test_paper["content"],
            title=test_paper["title"],
            document_id=test_paper["document_id"]
        )
        
        print("✅ Analysis Complete!")
        print(f"\n📊 **Results Summary:**")
        print(f"   Novel Contributions: {len(result.novel_contributions)}")
        print(f"   Research Quality Score: {result.research_quality_score:.2f}")
        print(f"   Technical Innovations: {len(result.technical_innovations)}")
        print(f"   Business Implications: {len(result.business_implications)}")
        
        if result.novel_contributions:
            print(f"\n🎯 **Sample Novel Contributions:**")
            for i, contribution in enumerate(result.novel_contributions[:2], 1):
                print(f"   {i}. {contribution}")
        
        # Show usage stats
        stats = analyzer.get_usage_statistics()
        print(f"\n📈 **Performance Stats:**")
        print(f"   Tokens Used: {stats['total_tokens']}")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        
        if "not available" in str(e):
            print(f"\n💡 To install this model:")
            print(f"   ollama pull {model_name}")
        
        return False


async def test_multi_model_comparison():
    """Test multiple models and compare their capabilities."""
    print("\n🔄 Multi-Model Comparison Test")
    print("=" * 40)
    
    # Check which models are available
    ollama_running, available_models = await check_ollama_status()
    
    if not ollama_running:
        print("❌ Ollama not running - cannot test models")
        return
    
    if not available_models:
        print("❌ No models installed - install a model first")
        return
    
    # Test available models
    test_models = []
    for model in available_models:
        if any(rec in model for rec in RECOMMENDED_MODELS.keys()):
            test_models.append(model)
    
    if not test_models:
        print("⚠️  No recommended models found in available models")
        print("💡 Available models:", available_models)
        print("💡 Recommended models:", list(RECOMMENDED_MODELS.keys()))
        # Use first available model for testing
        if available_models:
            test_models = [available_models[0]]
    
    print(f"🧪 Testing {len(test_models)} model(s): {test_models}")
    
    for model in test_models[:2]:  # Test max 2 models to avoid long runtime
        success = await test_local_model_analysis(model)
        if success:
            print(f"✅ {model} - Working!")
        else:
            print(f"❌ {model} - Failed")


def show_integration_summary():
    """Show summary of the Ollama integration."""
    print("\n" + "=" * 60)
    print("🎯 OLLAMA INTEGRATION SUMMARY")
    print("=" * 60)
    
    print("✅ **Integration Features:**")
    print("   • Ollama HTTP API integration")
    print("   • Automatic model availability checking")
    print("   • Token usage tracking")
    print("   • Multi-model support with easy swapping")
    print("   • Production-ready error handling")
    
    print("\n🧠 **Local Model Benefits:**")
    print("   • No API costs after hardware investment")
    print("   • Complete data privacy (no cloud calls)")
    print("   • Unlimited analysis capacity")
    print("   • Customizable model selection")
    print("   • Overnight batch processing capability")
    
    print("\n⚡ **Usage Examples:**")
    print("   # Lightweight testing")
    print("   analyzer = LLMAnalyzer(provider='ollama', model='llama3.1:8b')")
    print("   ")
    print("   # Production analysis")
    print("   analyzer = LLMAnalyzer(provider='ollama', model='llama3.1:70b')")
    print("   ")
    print("   # Technical papers")
    print("   analyzer = LLMAnalyzer(provider='ollama', model='deepseek-coder:33b')")
    
    print("\n🚀 **Next Steps:**")
    print("1. Install Ollama: https://ollama.ai/download")
    print("2. Pull recommended model: ollama pull llama3.1:8b")
    print("3. Test integration: python3 test_ollama_integration.py")
    print("4. Scale up to larger models as needed")


async def main():
    """Run all Ollama integration tests."""
    print("🚀 KnowHunt Ollama Local LLM Integration Test")
    print("=" * 60)
    print("Testing local model integration for cost-free, private research analysis\n")
    
    # Check Ollama status
    ollama_running, available_models = await check_ollama_status()
    
    # Show model recommendations
    await test_model_recommendations()
    
    # Test analysis if models are available
    if ollama_running and available_models:
        await test_multi_model_comparison()
    else:
        print("\n⚠️  Ollama setup required for live testing")
        print("📝 The integration code is ready - just need Ollama + models")
    
    # Show integration summary
    show_integration_summary()


if __name__ == "__main__":
    asyncio.run(main())