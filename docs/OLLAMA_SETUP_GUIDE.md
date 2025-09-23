# Ollama Local LLM Setup Guide for KnowHunt

This guide walks you through setting up local LLM models using Ollama for cost-free, private research analysis.

## 🎯 Overview

The KnowHunt system now supports local LLM analysis through Ollama integration, enabling:
- **Zero API costs** after initial hardware investment
- **Complete data privacy** (no cloud API calls)
- **Unlimited analysis capacity** 
- **Customizable model selection** for different analysis types
- **Overnight batch processing** capability

## 📋 Prerequisites

### Hardware Requirements

| Model Size | VRAM Required | Use Case |
|------------|---------------|----------|
| 7B-8B models | 6-8GB | Testing, lightweight analysis |
| 33B models | 20GB | Technical papers, code analysis |
| 70B+ models | 40GB+ | Production research analysis |

### Recommended Models

Based on the Phase 2 LLM strategy:

1. **llama3.1:8b** - Start here for testing
2. **llama3.1:70b** - Production analysis (if hardware allows)
3. **deepseek-coder:33b** - Technical/methods extraction
4. **qwen2.5:72b** - High-quality research writing
5. **mistral:7b** - Rapid prototyping

## 🚀 Installation Steps

### 1. Install Ollama

Visit [https://ollama.ai/download](https://ollama.ai/download) and download for your platform:

**Linux/WSL:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download the installer from the Ollama website.

### 2. Start Ollama Service

```bash
# Start Ollama (usually runs automatically after install)
ollama serve
```

Ollama will run on `http://localhost:11434` by default.

### 3. Pull Your First Model

Start with a lightweight model for testing:

```bash
# Recommended starter model (8GB VRAM)
ollama pull llama3.1:8b

# For more powerful analysis (40GB+ VRAM)
ollama pull llama3.1:70b

# For technical paper analysis (20GB VRAM)
ollama pull deepseek-coder:33b
```

### 4. Verify Installation

```bash
# Check available models
ollama list

# Test a simple query
ollama run llama3.1:8b "What is machine learning?"
```

## 🧪 Testing KnowHunt Integration

### Quick Test

```bash
cd /path/to/KnowHunt
python3 test_ollama_integration.py
```

### Manual Integration Test

```python
from knowhunt.analysis.llm_analyzer import LLMAnalyzer

# Initialize with local model
analyzer = LLMAnalyzer(
    provider="ollama", 
    model="llama3.1:8b"
)

# Test analysis
result = await analyzer.analyze_research_paper(
    text="Your research paper content...",
    title="Paper Title",
    document_id="test_001"
)

print(f"Novel contributions: {result.novel_contributions}")
```

## ⚙️ Configuration Options

### Model Selection Guide

```python
# Lightweight testing (6-8GB VRAM)
analyzer = LLMAnalyzer(provider="ollama", model="llama3.1:8b")

# Production analysis (40GB+ VRAM)
analyzer = LLMAnalyzer(provider="ollama", model="llama3.1:70b")

# Technical papers (20GB VRAM)
analyzer = LLMAnalyzer(provider="ollama", model="deepseek-coder:33b")

# Custom Ollama endpoint
analyzer = LLMAnalyzer(
    provider="ollama", 
    model="llama3.1:8b",
    ollama_base_url="http://192.168.1.100:11434"  # Remote Ollama server
)
```

### Performance Tuning

```python
# Optimize for speed
analyzer = LLMAnalyzer(
    provider="ollama", 
    model="llama3.1:8b",
    max_tokens=1000,      # Shorter responses
    temperature=0.1       # More focused outputs
)

# Optimize for quality
analyzer = LLMAnalyzer(
    provider="ollama", 
    model="llama3.1:70b",
    max_tokens=2500,      # Longer, detailed responses
    temperature=0.2       # Slightly more creative
)
```

## 🔄 Model Management

### Available Commands

```bash
# List installed models
ollama list

# Pull new model
ollama pull qwen2.5:72b

# Remove model to free space
ollama rm old-model:version

# Update existing model
ollama pull llama3.1:8b  # Re-pulls latest version

# Check model info
ollama show llama3.1:8b
```

### Storage Management

Models are stored in:
- **Linux/WSL:** `~/.ollama/models/`
- **macOS:** `~/.ollama/models/`
- **Windows:** `C:\Users\%username%\.ollama\models\`

Large models (70B) can use 40-80GB of disk space.

## 📊 Integration with KnowHunt Pipeline

### Basic Integration

```python
from knowhunt.analysis.llm_analyzer import LLMAnalyzer, LLMAnalysisService
from knowhunt.reports.llm_report_generator import LLMIntelligentReportGenerator

# Initialize local LLM service
analyzer = LLMAnalyzer(provider="ollama", model="llama3.1:8b")
llm_service = LLMAnalysisService(analyzer)

# Use in report generation
report_generator = LLMIntelligentReportGenerator(
    data_pipeline=your_pipeline,
    llm_analysis_service=llm_service
)
```

### Batch Processing Setup

For overnight analysis (Phase 2 strategy):

```python
# Set up for overnight processing
overnight_analyzer = LLMAnalyzer(
    provider="ollama", 
    model="llama3.1:70b",    # Use largest available model
    max_tokens=3000,         # Comprehensive analysis
    temperature=0.15         # Balanced creativity
)

# Process multiple papers
papers = get_collected_papers()
results = await llm_service.analyze_multiple_papers(papers)
```

## 🐛 Troubleshooting

### Common Issues

**1. "Cannot connect to host localhost:11434"**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if not running
ollama serve
```

**2. "Model not found"**
```bash
# Check available models
ollama list

# Pull the model if missing
ollama pull llama3.1:8b
```

**3. "Out of memory" errors**
```bash
# Check model size requirements
ollama show llama3.1:70b | grep "parameter size"

# Use smaller model if insufficient VRAM
ollama pull llama3.1:8b
```

**4. Slow response times**
```python
# Reduce max_tokens for faster responses
analyzer = LLMAnalyzer(
    provider="ollama", 
    model="llama3.1:8b",
    max_tokens=1000  # Reduce from default 2000
)
```

### Performance Optimization

**CPU Optimization:**
```bash
# Set thread count (adjust based on your CPU)
export OLLAMA_NUM_THREAD=8
ollama serve
```

**GPU Optimization:**
```bash
# Ensure GPU acceleration is working
nvidia-smi  # Should show Ollama process using GPU

# Check Ollama logs for GPU usage
ollama logs
```

## 🎯 Recommended Workflows

### Development Workflow
1. Start with `llama3.1:8b` for testing
2. Develop and debug your analysis prompts
3. Scale up to larger models for production

### Production Workflow
1. Use `llama3.1:70b` for comprehensive analysis
2. Set up overnight batch processing
3. Monitor token usage and performance

### Resource-Constrained Workflow
1. Use `mistral:7b` for basic analysis
2. Process papers in smaller batches
3. Focus on specific analysis types (e.g., just novel contributions)

## 🔗 Integration with Existing Pipeline

The Ollama integration is designed to work seamlessly with your existing KnowHunt pipeline:

```python
# Easy provider swapping - no other code changes needed
# analyzer = LLMAnalyzer(provider="openai", model="gpt-4")     # Cloud
analyzer = LLMAnalyzer(provider="ollama", model="llama3.1:8b") # Local

# Rest of analysis code remains identical
result = await analyzer.analyze_research_paper(...)
```

## 📈 Performance Expectations

| Model | Speed | Quality | VRAM | Use Case |
|-------|-------|---------|------|----------|
| mistral:7b | Fast | Good | 6GB | Quick testing |
| llama3.1:8b | Fast | Very Good | 8GB | Development |
| deepseek-coder:33b | Medium | Excellent* | 20GB | Technical papers |
| llama3.1:70b | Slow | Excellent | 40GB | Production |

*Especially for technical/code analysis

## 🎉 Success Indicators

Your integration is working when:
- ✅ `ollama list` shows your models
- ✅ `python3 test_ollama_integration.py` passes
- ✅ KnowHunt generates intelligent analysis results
- ✅ Reports show genuine research insights instead of keyword fragments

---

**Next Steps:** Once Ollama is set up, your KnowHunt system will have the intelligent research analysis capabilities needed to generate truly valuable reports with novel contribution detection, research significance assessment, and business intelligence insights.