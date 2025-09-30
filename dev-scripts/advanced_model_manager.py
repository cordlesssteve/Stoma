#!/usr/bin/env python3
"""
Advanced Model Manager for KnowHunt

This module manages various LLM models including large models like llama3.1:70b
and specialized models like deepseek-coder:33b, with automatic model selection
based on task requirements and system resources.
"""

import asyncio
import json
import logging
import subprocess
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import psutil
import platform

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    name: str
    size_gb: float
    vram_required_gb: float
    context_length: int
    specialty: str  # 'general', 'coding', 'reasoning', 'vision', 'scientific'
    description: str
    performance_tier: str  # 'lightweight', 'standard', 'advanced', 'premium'
    recommended_use_cases: List[str]
    estimated_speed: str  # 'fast', 'medium', 'slow'


class SystemResourceAnalyzer:
    """Analyzes system resources to recommend appropriate models."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            # Basic system info
            system_info = {
                "platform": platform.system(),
                "cpu_count": psutil.cpu_count(logical=True),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "disk_free_gb": psutil.disk_usage("/").free / (1024**3)
            }

            # Try to get GPU info
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_info = []

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info.append({
                        "name": name,
                        "memory_total_gb": memory.total / (1024**3),
                        "memory_free_gb": memory.free / (1024**3)
                    })

                system_info["gpu_info"] = gpu_info
                system_info["has_gpu"] = True

            except (ImportError, Exception):
                system_info["gpu_info"] = []
                system_info["has_gpu"] = False

            return system_info

        except Exception as e:
            logger.warning(f"Could not get complete system info: {e}")
            return {
                "platform": "unknown",
                "cpu_count": 4,
                "memory_gb": 8,
                "available_memory_gb": 4,
                "disk_free_gb": 50,
                "gpu_info": [],
                "has_gpu": False
            }

    @classmethod
    def recommend_models(cls, system_info: Dict[str, Any]) -> List[str]:
        """Recommend models based on system capabilities."""
        recommendations = []

        memory_gb = system_info.get("memory_gb", 8)
        has_gpu = system_info.get("has_gpu", False)
        gpu_memory = 0

        if has_gpu and system_info.get("gpu_info"):
            gpu_memory = max(gpu["memory_total_gb"] for gpu in system_info["gpu_info"])

        # Lightweight models (always recommended)
        recommendations.extend(["gemma2:2b", "qwen2.5:3b", "phi3:3.8b"])

        # Standard models (8GB+ system RAM)
        if memory_gb >= 12:
            recommendations.extend(["llama3.1:8b", "qwen2.5-coder:7b", "mistral:7b"])

        # Advanced models (16GB+ system RAM or GPU with 12GB+ VRAM)
        if memory_gb >= 16 or gpu_memory >= 12:
            recommendations.extend(["llama3.1:70b", "deepseek-coder:33b", "qwen2.5:32b"])

        # Premium models (32GB+ RAM or GPU with 24GB+ VRAM)
        if memory_gb >= 32 or gpu_memory >= 24:
            recommendations.extend(["llama3.1:405b", "qwen2.5:72b"])

        return recommendations


class AdvancedModelManager:
    """Manages advanced LLM models with automatic selection and deployment."""

    # Model registry with comprehensive information
    MODEL_REGISTRY = {
        # Lightweight Models (< 4GB)
        "gemma2:2b": ModelInfo(
            name="gemma2:2b",
            size_gb=2.2,
            vram_required_gb=3.0,
            context_length=8192,
            specialty="general",
            description="Google's lightweight, fast model for general tasks",
            performance_tier="lightweight",
            recommended_use_cases=["quick analysis", "testing", "batch processing"],
            estimated_speed="fast"
        ),
        "qwen2.5:3b": ModelInfo(
            name="qwen2.5:3b",
            size_gb=3.1,
            vram_required_gb=4.0,
            context_length=32768,
            specialty="general",
            description="Alibaba's efficient model with long context support",
            performance_tier="lightweight",
            recommended_use_cases=["long documents", "research analysis"],
            estimated_speed="fast"
        ),
        "phi3:3.8b": ModelInfo(
            name="phi3:3.8b",
            size_gb=3.8,
            vram_required_gb=5.0,
            context_length=4096,
            specialty="reasoning",
            description="Microsoft's reasoning-focused compact model",
            performance_tier="lightweight",
            recommended_use_cases=["logical reasoning", "problem solving"],
            estimated_speed="fast"
        ),

        # Standard Models (4-12GB)
        "llama3.1:8b": ModelInfo(
            name="llama3.1:8b",
            size_gb=8.0,
            vram_required_gb=10.0,
            context_length=131072,
            specialty="general",
            description="Meta's balanced model for comprehensive analysis",
            performance_tier="standard",
            recommended_use_cases=["research analysis", "content generation", "reasoning"],
            estimated_speed="medium"
        ),
        "qwen2.5-coder:7b": ModelInfo(
            name="qwen2.5-coder:7b",
            size_gb=7.5,
            vram_required_gb=9.0,
            context_length=32768,
            specialty="coding",
            description="Alibaba's code-specialized model",
            performance_tier="standard",
            recommended_use_cases=["code analysis", "technical papers", "software research"],
            estimated_speed="medium"
        ),
        "mistral:7b": ModelInfo(
            name="mistral:7b",
            size_gb=7.2,
            vram_required_gb=9.0,
            context_length=32768,
            specialty="general",
            description="Mistral AI's efficient general-purpose model",
            performance_tier="standard",
            recommended_use_cases=["analysis", "summarization", "research"],
            estimated_speed="medium"
        ),

        # Advanced Models (12-40GB)
        "llama3.1:70b": ModelInfo(
            name="llama3.1:70b",
            size_gb=40.0,
            vram_required_gb=42.0,
            context_length=131072,
            specialty="general",
            description="Meta's large model for complex reasoning and analysis",
            performance_tier="advanced",
            recommended_use_cases=["complex research", "detailed analysis", "synthesis"],
            estimated_speed="slow"
        ),
        "deepseek-coder:33b": ModelInfo(
            name="deepseek-coder:33b",
            size_gb=20.0,
            vram_required_gb=22.0,
            context_length=16384,
            specialty="coding",
            description="DeepSeek's large coding model for technical analysis",
            performance_tier="advanced",
            recommended_use_cases=["technical papers", "code analysis", "software research"],
            estimated_speed="slow"
        ),
        "qwen2.5:32b": ModelInfo(
            name="qwen2.5:32b",
            size_gb=19.0,
            vram_required_gb=21.0,
            context_length=32768,
            specialty="general",
            description="Alibaba's large model with extended context",
            performance_tier="advanced",
            recommended_use_cases=["complex analysis", "long documents", "research synthesis"],
            estimated_speed="slow"
        ),

        # Premium Models (40GB+)
        "qwen2.5:72b": ModelInfo(
            name="qwen2.5:72b",
            size_gb=43.0,
            vram_required_gb=45.0,
            context_length=32768,
            specialty="general",
            description="Alibaba's flagship large model",
            performance_tier="premium",
            recommended_use_cases=["highest quality analysis", "complex reasoning", "research synthesis"],
            estimated_speed="very_slow"
        )
    }

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.system_info = SystemResourceAnalyzer.get_system_info()
        self.recommended_models = SystemResourceAnalyzer.recommend_models(self.system_info)

    async def get_available_models(self) -> List[str]:
        """Get list of currently available models."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['name'] for model in data.get('models', [])]
                    else:
                        return []
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    async def install_model(self, model_name: str, show_progress: bool = True) -> bool:
        """Install a model with progress tracking."""
        try:
            logger.info(f"Installing model: {model_name}")

            if show_progress:
                print(f"📥 Downloading {model_name}...")
                if model_name in self.MODEL_REGISTRY:
                    model_info = self.MODEL_REGISTRY[model_name]
                    print(f"   Size: {model_info.size_gb:.1f}GB")
                    print(f"   VRAM Required: {model_info.vram_required_gb:.1f}GB")
                    print(f"   This may take several minutes...")

            # Run ollama pull command
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=1800)  # 30 minute timeout

                if process.returncode == 0:
                    logger.info(f"Successfully installed {model_name}")
                    if show_progress:
                        print(f"✅ {model_name} installed successfully")
                    return True
                else:
                    logger.error(f"Failed to install {model_name}: {stderr}")
                    if show_progress:
                        print(f"❌ Failed to install {model_name}: {stderr}")
                    return False

            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"Installation of {model_name} timed out")
                if show_progress:
                    print(f"❌ Installation timed out for {model_name}")
                return False

        except Exception as e:
            logger.error(f"Error installing {model_name}: {e}")
            return False

    def select_model_for_task(self, task_type: str, content_length: int = 1000,
                            performance_preference: str = "balanced") -> str:
        """Select optimal model based on task requirements."""

        # Get available models
        available_models = []
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                available_models = [line.split()[0] for line in lines if line.strip()]
        except Exception:
            available_models = []

        if not available_models:
            return "gemma2:2b"  # Fallback

        # Filter models by task type
        suitable_models = []

        for model_name in available_models:
            if model_name in self.MODEL_REGISTRY:
                model_info = self.MODEL_REGISTRY[model_name]

                # Check if model is suitable for task
                task_match = False

                if task_type in ["general", "research_analysis", "summarization"]:
                    if model_info.specialty in ["general", "reasoning"]:
                        task_match = True
                elif task_type in ["code_analysis", "technical_papers"]:
                    if model_info.specialty in ["coding", "general"]:
                        task_match = True
                elif task_type == "synthesis":
                    if model_info.performance_tier in ["advanced", "premium"]:
                        task_match = True
                else:
                    task_match = True  # Any model for unknown tasks

                if task_match:
                    suitable_models.append((model_name, model_info))

        if not suitable_models:
            return available_models[0]  # Return any available model

        # Select based on performance preference and content length
        if performance_preference == "speed":
            # Prefer fast models
            suitable_models.sort(key=lambda x: (x[1].estimated_speed == "fast", -x[1].size_gb))
        elif performance_preference == "quality":
            # Prefer large, high-quality models
            suitable_models.sort(key=lambda x: -x[1].size_gb)
        else:  # balanced
            # Balance size and context length for content
            def balance_score(model_tuple):
                _, model_info = model_tuple
                size_penalty = model_info.size_gb / 50  # Normalize size
                context_bonus = min(model_info.context_length / content_length, 2.0)
                return context_bonus - size_penalty

            suitable_models.sort(key=balance_score, reverse=True)

        return suitable_models[0][0]

    async def setup_recommended_models(self, max_models: int = 3,
                                     include_advanced: bool = None) -> List[str]:
        """Set up recommended models based on system capabilities."""

        if include_advanced is None:
            # Auto-decide based on system resources
            memory_gb = self.system_info.get("memory_gb", 8)
            include_advanced = memory_gb >= 16

        available_models = await self.get_available_models()
        installed_models = []

        # Prioritize models to install
        priority_models = []

        # Always include lightweight models
        priority_models.extend(["gemma2:2b", "qwen2.5:3b"])

        # Add standard models if system can handle them
        if self.system_info.get("memory_gb", 8) >= 12:
            priority_models.extend(["llama3.1:8b", "qwen2.5-coder:7b"])

        # Add advanced models if requested and system can handle them
        if include_advanced and self.system_info.get("memory_gb", 8) >= 16:
            priority_models.extend(["llama3.1:70b", "deepseek-coder:33b"])

        # Install missing models up to max_models
        install_count = 0
        for model_name in priority_models:
            if install_count >= max_models:
                break

            if model_name not in available_models:
                print(f"\n🔄 Installing {model_name}...")
                success = await self.install_model(model_name, show_progress=True)
                if success:
                    installed_models.append(model_name)
                    install_count += 1
                else:
                    logger.warning(f"Failed to install {model_name}")

        return installed_models

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a model."""
        return self.MODEL_REGISTRY.get(model_name)

    def print_system_analysis(self):
        """Print system analysis and model recommendations."""
        print("🖥️  System Analysis")
        print("=" * 30)
        print(f"Platform: {self.system_info.get('platform', 'Unknown')}")
        print(f"CPU Cores: {self.system_info.get('cpu_count', 'Unknown')}")
        print(f"RAM: {self.system_info.get('memory_gb', 0):.1f} GB")
        print(f"Available RAM: {self.system_info.get('available_memory_gb', 0):.1f} GB")
        print(f"GPU Available: {'Yes' if self.system_info.get('has_gpu') else 'No'}")

        if self.system_info.get('gpu_info'):
            print("\n🔥 GPU Information:")
            for i, gpu in enumerate(self.system_info['gpu_info']):
                print(f"  GPU {i+1}: {gpu['name']}")
                print(f"    VRAM: {gpu['memory_total_gb']:.1f} GB")

        print(f"\n💡 Recommended Models:")
        for model in self.recommended_models:
            if model in self.MODEL_REGISTRY:
                info = self.MODEL_REGISTRY[model]
                print(f"  - {model} ({info.performance_tier}, {info.size_gb:.1f}GB)")

    def get_model_compatibility_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Get compatibility matrix showing which models work with current system."""
        memory_gb = self.system_info.get("memory_gb", 8)
        gpu_memory = 0

        if self.system_info.get("gpu_info"):
            gpu_memory = max(gpu["memory_total_gb"] for gpu in self.system_info["gpu_info"])

        compatibility = {}

        for model_name, model_info in self.MODEL_REGISTRY.items():
            # Determine compatibility
            can_run_cpu = memory_gb >= model_info.size_gb * 1.2  # 20% overhead
            can_run_gpu = gpu_memory >= model_info.vram_required_gb

            compatibility[model_name] = {
                "model_info": model_info,
                "can_run_cpu": can_run_cpu,
                "can_run_gpu": can_run_gpu,
                "recommended": model_name in self.recommended_models,
                "resource_usage": {
                    "cpu_memory_gb": model_info.size_gb * 1.2,
                    "gpu_memory_gb": model_info.vram_required_gb
                }
            }

        return compatibility


async def main():
    """Main CLI for advanced model management."""
    import sys

    if len(sys.argv) < 2:
        print("Advanced Model Manager for KnowHunt")
        print("Usage: python3 advanced_model_manager.py <command> [options]")
        print("\nCommands:")
        print("  analyze           - Analyze system and show recommendations")
        print("  install <model>   - Install a specific model")
        print("  setup             - Install recommended models")
        print("  compatibility     - Show model compatibility matrix")
        print("  select <task>     - Select best model for task type")
        print("\nExamples:")
        print("  python3 advanced_model_manager.py analyze")
        print("  python3 advanced_model_manager.py install llama3.1:70b")
        print("  python3 advanced_model_manager.py setup")
        print("  python3 advanced_model_manager.py select code_analysis")
        return

    manager = AdvancedModelManager()
    command = sys.argv[1].lower()

    if command == "analyze":
        manager.print_system_analysis()

    elif command == "install":
        if len(sys.argv) < 3:
            print("Usage: python3 advanced_model_manager.py install <model_name>")
            return

        model_name = sys.argv[2]
        success = await manager.install_model(model_name)
        if success:
            print(f"✅ Successfully installed {model_name}")
        else:
            print(f"❌ Failed to install {model_name}")

    elif command == "setup":
        print("🚀 Setting up recommended models...")
        manager.print_system_analysis()

        include_advanced = input("\nInstall advanced models (llama3.1:70b, deepseek-coder:33b)? [y/N]: ").lower().startswith('y')
        max_models = int(input("Maximum models to install [3]: ") or 3)

        installed = await manager.setup_recommended_models(max_models=max_models, include_advanced=include_advanced)

        print(f"\n✅ Setup complete! Installed {len(installed)} models:")
        for model in installed:
            print(f"  - {model}")

    elif command == "compatibility":
        print("📊 Model Compatibility Matrix")
        print("=" * 40)

        compatibility = manager.get_model_compatibility_matrix()

        for model_name, info in compatibility.items():
            model_info = info["model_info"]
            status = "✅" if info["recommended"] else "⚠️" if info["can_run_cpu"] or info["can_run_gpu"] else "❌"

            print(f"\n{status} {model_name} ({model_info.performance_tier})")
            print(f"    Size: {model_info.size_gb:.1f}GB, VRAM: {model_info.vram_required_gb:.1f}GB")
            print(f"    Can run on CPU: {'Yes' if info['can_run_cpu'] else 'No'}")
            print(f"    Can run on GPU: {'Yes' if info['can_run_gpu'] else 'No'}")
            print(f"    Specialty: {model_info.specialty}")

    elif command == "select":
        if len(sys.argv) < 3:
            print("Usage: python3 advanced_model_manager.py select <task_type>")
            print("Task types: general, code_analysis, technical_papers, synthesis, research_analysis")
            return

        task_type = sys.argv[2]
        selected_model = manager.select_model_for_task(task_type)

        print(f"🎯 Selected model for '{task_type}': {selected_model}")

        if selected_model in manager.MODEL_REGISTRY:
            info = manager.MODEL_REGISTRY[selected_model]
            print(f"   Description: {info.description}")
            print(f"   Performance: {info.performance_tier}")
            print(f"   Speed: {info.estimated_speed}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    asyncio.run(main())