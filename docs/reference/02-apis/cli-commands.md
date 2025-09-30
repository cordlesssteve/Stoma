# Stoma CLI Commands Reference

**Status**: ACTIVE
**Last Updated**: 2025-09-29
**Version**: 1.0

## Overview

Stoma provides a comprehensive CLI interface for research intelligence, data collection, analysis, and reporting. This document covers all available commands and their usage.

## Core Commands

### Collection Commands

#### `stoma collect-arxiv`
Collect papers from ArXiv with content enrichment.

```bash
stoma collect-arxiv -q "machine learning" -n 5 -o results.json
```

**Parameters:**
- `-q, --query`: Search query for ArXiv papers
- `-n, --number`: Number of papers to collect (default: 10)
- `-o, --output`: Output file path for results

#### `stoma collect-reddit`
Collect discussions from Reddit with content enrichment.

```bash
stoma collect-reddit -s "MachineLearning" -n 20 -o reddit_data.json
```

### LLM Analysis Commands

#### `stoma llm analyze-text`
Perform intelligent analysis on provided text using LLM capabilities.

```bash
stoma llm analyze-text "research text here" -o analysis.json
stoma llm analyze-text "research text" --provider ollama --model llama3.1:latest
```

**Parameters:**
- `text`: Text content to analyze
- `-o, --output`: Output file for analysis results
- `--provider`: LLM provider (openai, anthropic, ollama)
- `--model`: Specific model to use

#### `stoma llm collect-and-analyze-arxiv`
End-to-end workflow: collect papers and perform LLM analysis.

```bash
stoma llm collect-and-analyze-arxiv -q "protein folding" -n 3
```

**Parameters:**
- `-q, --query`: ArXiv search query
- `-n, --number`: Number of papers to analyze
- `--provider`: LLM provider for analysis
- `--model`: Specific model for analysis

#### `stoma llm test-providers`
Test availability and configuration of LLM providers.

```bash
stoma llm test-providers
```

### Deep Research Commands (NEW)

#### `stoma deep-research ollama-research`
Perform comprehensive research using OpenDeepResearch multi-agent architecture.

```bash
stoma deep-research ollama-research "reinforcement learning architectures" --model llama3.1:latest
```

**Parameters:**
- `topic`: Research topic or question
- `--model`: Ollama model to use (default: llama3.1:latest)
- `--output`: Output file for research report
- `--context`: Additional context for research

#### `stoma deep-research analyze-papers`
Multi-paper analysis workflow using deep research capabilities.

```bash
stoma deep-research analyze-papers "papers.json" --model llama3.1:latest
```

#### `stoma deep-research citation-research`
Research with automatic citation collection and formatting.

```bash
stoma deep-research citation-research "semantic reasoning AI" --model llama3.1:latest
```

### Report Management Commands

#### `stoma llm search-reports`
Search through stored analysis reports.

```bash
stoma llm search-reports --query "protein folding" --min-quality 5
```

**Parameters:**
- `--query`: Search query for report content
- `--min-quality`: Minimum quality threshold
- `--provider`: Filter by LLM provider
- `--date-from`: Filter reports from date
- `--date-to`: Filter reports to date

#### `stoma llm view-report`
View a specific analysis report.

```bash
stoma llm view-report cli_analysis_20250924_152955
```

**Parameters:**
- `report_id`: Unique identifier for the report

#### `stoma llm storage-stats`
View statistics about stored reports and storage usage.

```bash
stoma llm storage-stats
```

### NLP Analysis Commands

#### `stoma nlp batch-analyze`
Perform traditional NLP analysis on collected content.

```bash
stoma nlp batch-analyze --limit 100
```

#### `stoma nlp detect-trends`
Detect trends and patterns in collected content.

```bash
stoma nlp detect-trends --timeframe 30
```

## Global Options

All commands support these global options:

- `--help`: Show help for the command
- `--verbose`: Enable verbose output
- `--config`: Specify custom configuration file
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Configuration

### Environment Variables

```bash
# LLM Provider API Keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Database Configuration
export DATABASE_URL="postgresql://user:pass@localhost:5433/stoma"

# Ollama Configuration
export OLLAMA_BASE_URL="http://localhost:11434"
```

### Model Selection Guidelines

#### Ollama Models (Local)
- `gemma2:2b` (2.2GB) - Quick analysis, testing
- `llama3.1:8b` (8GB) - Standard research papers
- `llama3.1:70b` (40GB) - Complex research, production
- `deepseek-coder:33b` (20GB) - Technical papers
- `qwen2.5:72b` (40GB) - Research writing

#### Cloud Models
- `gpt-4` - High-quality analysis (OpenAI)
- `claude-3-sonnet` - Balanced performance (Anthropic)
- `gpt-3.5-turbo` - Cost-effective analysis (OpenAI)

## Error Handling

### Common Issues

1. **Provider Not Available**
   ```
   Error: OpenAI provider not configured
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **Ollama Connection Failed**
   ```
   Error: Could not connect to Ollama
   Solution: Start Ollama service with 'ollama serve'
   ```

3. **Model Not Found**
   ```
   Error: Model llama3.1:latest not found
   Solution: Pull model with 'ollama pull llama3.1:latest'
   ```

### Debug Mode

Enable debug mode for detailed error information:

```bash
stoma --log-level DEBUG llm analyze-text "test"
```

## Examples

### Research Workflow Example

```bash
# 1. Collect papers from ArXiv
stoma collect-arxiv -q "transformer attention mechanisms" -n 5 -o papers.json

# 2. Perform deep research analysis
stoma deep-research ollama-research "How have attention mechanisms evolved in transformer architectures?" --model llama3.1:latest

# 3. Search previous reports
stoma llm search-reports --query "attention mechanisms" --min-quality 7

# 4. View specific report
stoma llm view-report analysis_20250929_134937
```

### Local vs Cloud Analysis

```bash
# Using local Ollama model (free)
stoma llm analyze-text "research content" --provider ollama --model llama3.1:latest

# Using cloud model (API cost)
stoma llm analyze-text "research content" --provider openai --model gpt-4
```