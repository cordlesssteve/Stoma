#!/usr/bin/env python3
"""
Lightweight LLM configuration for KnowHunt production.

Optimized for memory efficiency and cost-effectiveness while providing
significantly better analysis than keyword extraction.
"""

import logging
from typing import Dict, Any, Optional
from knowhunt.analysis.llm_analyzer import LLMAnalyzer

logger = logging.getLogger(__name__)


class LightweightLLMConfig:
    """Configuration manager for lightweight LLM models."""
    
    # Recommended lightweight models in order of preference
    LIGHTWEIGHT_MODELS = [
        {
            "name": "phi3.5",
            "description": "Microsoft Phi-3.5 (3.8B params)",
            "memory_gb": 3,
            "quality_score": 8.0,
            "speed_score": 9.0,
            "use_case": "Best overall balance"
        },
        {
            "name": "qwen2.5:3b", 
            "description": "Qwen2.5-3B-Instruct",
            "memory_gb": 3,
            "quality_score": 8.0,
            "speed_score": 8.5,
            "use_case": "Excellent for research"
        },
        {
            "name": "llama3.2:3b",
            "description": "Meta Llama-3.2-3B",
            "memory_gb": 3,
            "quality_score": 8.2,
            "speed_score": 8.0,
            "use_case": "Latest small model"
        },
        {
            "name": "gemma2:2b",
            "description": "Google Gemma-2-2B", 
            "memory_gb": 2,
            "quality_score": 6.5,
            "speed_score": 9.5,
            "use_case": "Ultra-fast summaries"
        },
        {
            "name": "qwen2.5:7b",
            "description": "Qwen2.5-7B-Instruct",
            "memory_gb": 6,
            "quality_score": 8.5,
            "speed_score": 7.0,
            "use_case": "Higher quality analysis"
        },
        {
            "name": "mistral",
            "description": "Mistral-7B-Instruct",
            "memory_gb": 5,
            "quality_score": 8.0,
            "speed_score": 7.5,
            "use_case": "Balanced performance"
        }
    ]
    
    @classmethod
    def get_recommended_model(cls, 
                            memory_limit_gb: Optional[float] = None,
                            prefer_speed: bool = False,
                            prefer_quality: bool = False) -> Dict[str, Any]:
        """
        Get the best recommended model based on constraints.
        
        Args:
            memory_limit_gb: Maximum memory to use (None for no limit)
            prefer_speed: Prioritize response speed
            prefer_quality: Prioritize analysis quality
            
        Returns:
            Model configuration dict
        """
        available_models = cls.LIGHTWEIGHT_MODELS.copy()
        
        # Filter by memory limit
        if memory_limit_gb:
            available_models = [
                m for m in available_models 
                if m["memory_gb"] <= memory_limit_gb
            ]
        
        if not available_models:
            # Fallback to smallest model
            return cls.LIGHTWEIGHT_MODELS[-2]  # gemma2:2b
        
        # Sort by preference
        if prefer_speed:
            available_models.sort(key=lambda x: x["speed_score"], reverse=True)
        elif prefer_quality:
            available_models.sort(key=lambda x: x["quality_score"], reverse=True)
        else:
            # Default: balance of quality and speed
            available_models.sort(
                key=lambda x: (x["quality_score"] + x["speed_score"]) / 2, 
                reverse=True
            )
        
        return available_models[0]
    
    @classmethod
    def create_lightweight_analyzer(cls,
                                  memory_limit_gb: Optional[float] = None,
                                  prefer_speed: bool = False,
                                  prefer_quality: bool = False,
                                  fallback_to_cloud: bool = True) -> LLMAnalyzer:
        """
        Create an optimized lightweight analyzer.
        
        Args:
            memory_limit_gb: Maximum memory constraint
            prefer_speed: Optimize for speed over quality
            prefer_quality: Optimize for quality over speed  
            fallback_to_cloud: Fall back to cloud models if local unavailable
            
        Returns:
            Configured LLMAnalyzer instance
        """
        model_config = cls.get_recommended_model(
            memory_limit_gb=memory_limit_gb,
            prefer_speed=prefer_speed,
            prefer_quality=prefer_quality
        )
        
        logger.info(f"Selected lightweight model: {model_config['name']} "
                   f"({model_config['description']})")
        
        # Configure for lightweight usage
        analyzer_config = {
            "provider": "ollama",
            "model": model_config["name"],
            "max_tokens": 800 if prefer_speed else 1200,  # Adjust for speed vs quality
            "temperature": 0.0 if prefer_quality else 0.1  # Lower for consistency
        }
        
        try:
            analyzer = LLMAnalyzer(**analyzer_config)
            logger.info(f"Initialized lightweight analyzer with {model_config['name']}")
            return analyzer
            
        except Exception as e:
            logger.warning(f"Failed to initialize local model: {e}")
            
            if fallback_to_cloud:
                logger.info("Falling back to cloud model...")
                # Fallback to cloud with lightweight settings
                return LLMAnalyzer(
                    provider="openai",  # or "anthropic"
                    model="gpt-3.5-turbo",  # Cheaper than GPT-4
                    max_tokens=800,
                    temperature=0.1
                )
            else:
                raise


class ProductionLightweightConfig:
    """Production configuration for lightweight analysis pipeline."""
    
    def __init__(self, 
                 memory_budget_gb: float = 4.0,
                 analysis_mode: str = "balanced"):
        """
        Initialize production configuration.
        
        Args:
            memory_budget_gb: Memory budget for LLM models
            analysis_mode: "fast", "balanced", or "quality"
        """
        self.memory_budget = memory_budget_gb
        self.analysis_mode = analysis_mode
        
        # Configure based on mode
        mode_configs = {
            "fast": {"prefer_speed": True, "prefer_quality": False},
            "quality": {"prefer_speed": False, "prefer_quality": True}, 
            "balanced": {"prefer_speed": False, "prefer_quality": False}
        }
        
        self.config = mode_configs.get(analysis_mode, mode_configs["balanced"])
        self.analyzer = None
    
    async def initialize_analyzer(self) -> LLMAnalyzer:
        """Initialize the lightweight analyzer for production use."""
        if self.analyzer is None:
            self.analyzer = LightweightLLMConfig.create_lightweight_analyzer(
                memory_limit_gb=self.memory_budget,
                **self.config,
                fallback_to_cloud=True  # Always allow cloud fallback in production
            )
        
        return self.analyzer
    
    async def analyze_paper_batch(self, papers: list) -> list:
        """
        Analyze a batch of papers with lightweight models.
        
        Optimized for throughput and resource efficiency.
        """
        analyzer = await self.initialize_analyzer()
        results = []
        
        for i, paper in enumerate(papers):
            try:
                logger.info(f"Analyzing paper {i+1}/{len(papers)}: {paper.get('title', 'Untitled')}")
                
                result = await analyzer.analyze_research_paper(
                    text=paper.get("content", ""),
                    title=paper.get("title", ""),
                    document_id=paper.get("id", f"paper_{i}")
                )
                
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    stats = analyzer.get_usage_statistics()
                    logger.info(f"Processed {i+1} papers. Success rate: {stats['success_rate']:.1%}")
                    
            except Exception as e:
                logger.error(f"Failed to analyze paper {i+1}: {e}")
                # Continue with next paper rather than failing entire batch
                continue
        
        return results
    
    def get_analysis_cost_estimate(self, num_papers: int) -> Dict[str, Any]:
        """
        Estimate analysis costs for lightweight vs cloud models.
        
        Args:
            num_papers: Number of papers to analyze
            
        Returns:
            Cost comparison data
        """
        avg_tokens_per_paper = 1000  # Conservative estimate
        
        return {
            "lightweight_local": {
                "setup_cost_usd": 200,  # GPU for lightweight models
                "per_paper_cost_usd": 0.0,  # No ongoing costs
                "total_cost_usd": 200,
                "break_even_papers": 4000,  # Break even vs cloud at ~$0.05/paper
                "memory_required_gb": self.memory_budget
            },
            "cloud_gpt35": {
                "setup_cost_usd": 0,
                "per_paper_cost_usd": 0.002,  # ~$0.002 per paper
                "total_cost_usd": num_papers * 0.002,
                "break_even_papers": 0,
                "memory_required_gb": 0
            },
            "cloud_gpt4": {
                "setup_cost_usd": 0,
                "per_paper_cost_usd": 0.05,  # ~$0.05 per paper  
                "total_cost_usd": num_papers * 0.05,
                "break_even_papers": 0,
                "memory_required_gb": 0
            },
            "recommendation": (
                "lightweight_local" if num_papers > 1000 else "cloud_gpt35"
            )
        }


# Convenience functions for quick setup
def create_fast_analyzer() -> LLMAnalyzer:
    """Create analyzer optimized for speed (2-3GB memory)."""
    return LightweightLLMConfig.create_lightweight_analyzer(
        memory_limit_gb=3.0,
        prefer_speed=True
    )


def create_quality_analyzer() -> LLMAnalyzer:
    """Create analyzer optimized for quality (up to 6GB memory)."""
    return LightweightLLMConfig.create_lightweight_analyzer(
        memory_limit_gb=6.0,
        prefer_quality=True
    )


def create_budget_analyzer() -> LLMAnalyzer:
    """Create ultra-lightweight analyzer (2GB memory)."""
    return LightweightLLMConfig.create_lightweight_analyzer(
        memory_limit_gb=2.0,
        prefer_speed=True
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🚀 Lightweight LLM Configuration Demo")
        print("=" * 40)
        
        # Show model recommendations
        print("\n📋 Model Recommendations:")
        for constraint in [2, 3, 6]:
            model = LightweightLLMConfig.get_recommended_model(memory_limit_gb=constraint)
            print(f"   {constraint}GB limit: {model['name']} ({model['description']})")
        
        # Show cost analysis
        config = ProductionLightweightConfig(memory_budget_gb=3.0)
        costs = config.get_analysis_cost_estimate(1000)
        print(f"\n💰 Cost Analysis for 1000 papers:")
        print(f"   Lightweight local: ${costs['lightweight_local']['total_cost_usd']}")
        print(f"   Cloud GPT-3.5: ${costs['cloud_gpt35']['total_cost_usd']}")
        print(f"   Recommendation: {costs['recommendation']}")
    
    asyncio.run(demo())