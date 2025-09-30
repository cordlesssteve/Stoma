#!/usr/bin/env python3
"""
Simple test using Ollama directly for reinforcement learning analysis.
"""

import asyncio
import json
from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


async def simple_ollama_analysis():
    """Simple analysis using Ollama directly."""

    print("🚀 Simple Ollama Analysis: Reinforcement Learning Architectures")
    print("=" * 60)

    # Initialize Ollama model
    try:
        model = init_chat_model(
            model="ollama:llama3.1:latest",
            max_tokens=4096,
            temperature=0.1
        )
        print("✅ Model initialized: llama3.1:latest")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

    # Research prompt
    prompt = """You are a research expert specializing in machine learning. Please provide a comprehensive analysis of reinforcement learning architectures.

Your analysis should cover:

1. **Current State of the Field**: Overview of where RL architectures stand today

2. **Key Architectural Approaches**:
   - Value-based methods (DQN, Double DQN, Dueling DQN)
   - Policy-based methods (REINFORCE, A2C, A3C)
   - Actor-Critic methods (PPO, SAC, TD3)
   - Model-based approaches

3. **Recent Innovations**:
   - Transformer-based RL (Decision Transformer, Trajectory Transformer)
   - Multi-agent architectures
   - Hierarchical RL structures
   - Meta-learning approaches

4. **Technical Strengths and Limitations**: Detailed analysis of trade-offs

5. **Future Research Directions**: Emerging trends and open challenges

6. **Practical Applications**: Where these architectures excel in real-world scenarios

Please provide a technical, in-depth analysis with specific details about architectural components, training mechanisms, and performance characteristics."""

    print("🔄 Running analysis (this may take 2-3 minutes)...")
    start_time = datetime.now()

    try:
        # Run the analysis
        response = await model.ainvoke([HumanMessage(content=prompt)])

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"✅ Analysis completed in {duration.total_seconds():.1f} seconds")
        print("=" * 60)
        print("📋 REINFORCEMENT LEARNING ARCHITECTURES ANALYSIS")
        print("=" * 60)
        print(response.content)
        print("=" * 60)

        # Save results
        result = {
            "topic": "reinforcement learning architectures",
            "analysis": response.content,
            "model": "llama3.1:latest",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration.total_seconds(),
            "prompt_used": prompt,
            "success": True
        }

        output_file = f"rl_analysis_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"💾 Full analysis saved to: {output_file}")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(simple_ollama_analysis())
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("Your local Ollama setup is working perfectly for deep research!")
    else:
        print("\n💥 Analysis failed. Check the error messages above.")