# Stoma Testing Guide

**Status**: ACTIVE
**Last Updated**: 2025-09-29
**Version**: 1.0

## Overview

This guide covers testing procedures for Stoma's research intelligence system, including integration tests, unit tests, and development workflows.

## Test Structure

```
tests/
├── integration/              # Integration tests
│   ├── test_enhanced_pipeline.py
│   ├── test_llm_intelligence.py
│   ├── test_ollama_integration.py
│   └── test_deep_research_integration.py
├── unit/                    # Unit tests
└── fixtures/                # Test data and fixtures
```

## Integration Tests

### LLM Analysis Tests

**File**: `tests/integration/test_llm_intelligence.py`

```bash
# Test LLM analysis capabilities
python3 tests/integration/test_llm_intelligence.py
```

**Coverage**:
- Multi-provider LLM integration (OpenAI, Anthropic, Ollama)
- Analysis quality validation
- Report generation and storage
- Error handling and fallback mechanisms

### Ollama Integration Tests

**File**: `tests/integration/test_ollama_integration.py`

```bash
# Test local Ollama model integration
python3 tests/integration/test_ollama_integration.py
```

**Coverage**:
- Ollama service connectivity
- Model availability and loading
- Local analysis workflows
- Performance benchmarking

### OpenDeepResearch Integration Tests

**File**: `tests/integration/test_deep_research_integration.py`

```bash
# Test OpenDeepResearch multi-agent workflows
python3 tests/integration/test_deep_research_integration.py
```

**Coverage**:
- Git submodule integration
- Multi-agent coordination
- Tool calling functionality
- Report assembly and storage

### Enhanced Pipeline Tests

**File**: `tests/integration/test_enhanced_pipeline.py`

```bash
# Test complete content enrichment pipeline
python3 tests/integration/test_enhanced_pipeline.py
```

**Coverage**:
- Content collection and enrichment
- 48.5x improvement validation
- PDF extraction and web scraping
- End-to-end workflow testing

## Test Commands

### Core Testing Commands

```bash
# Test enhanced pipeline with content enrichment
python3 tests/integration/test_enhanced_pipeline.py

# Test real pipeline with ArXiv papers
python3 tests/integration/test_real_pipeline.py

# Test LLM intelligence integration
python3 tests/integration/test_llm_intelligence.py

# Test Ollama local model integration
python3 tests/integration/test_ollama_integration.py
```

### CLI Testing

```bash
# Test CLI commands
stoma llm test-providers
stoma deep-research ollama-research "test topic" --model llama3.1:latest
```

## Development Testing Workflow

### 1. Pre-commit Testing

```bash
# Run before committing changes
python3 tests/integration/test_enhanced_pipeline.py
python3 tests/integration/test_llm_intelligence.py
```

### 2. Feature Testing

```bash
# Test specific features
python3 tests/integration/test_ollama_integration.py  # Local models
python3 tests/integration/test_deep_research_integration.py  # Multi-agent
```

### 3. Performance Testing

```bash
# Benchmark content enrichment
python3 tests/integration/test_enhanced_pipeline.py --benchmark

# Benchmark LLM analysis
python3 tests/integration/test_llm_intelligence.py --benchmark
```

## Test Configuration

### Environment Setup

```bash
# Required environment variables
export OPENAI_API_KEY="your-key-here"  # For cloud LLM testing
export ANTHROPIC_API_KEY="your-key-here"  # For cloud LLM testing

# Optional configuration
export OLLAMA_BASE_URL="http://localhost:11434"  # Local Ollama
export DATABASE_URL="postgresql://user:pass@localhost:5433/stoma"
```

### Test Data

Test data is stored in `tests/fixtures/`:
- Sample research papers
- Mock API responses
- Expected analysis outputs
- Performance benchmarks

## Success Criteria

### Content Enrichment
- 40x+ content improvement ratio
- PDF extraction success rate >90%
- Web scraping robots.txt compliance

### LLM Analysis
- Novel contribution detection accuracy
- Research significance scoring consistency
- Business intelligence extraction quality

### Integration
- OpenDeepResearch multi-agent coordination
- Tool calling success with Ollama models
- Report assembly and storage functionality

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```
   Error: Could not connect to Ollama
   Solution: Start Ollama with 'ollama serve'
   ```

2. **Model Not Found**
   ```
   Error: Model not found
   Solution: Pull model with 'ollama pull llama3.1:latest'
   ```

3. **API Key Missing**
   ```
   Error: Provider not configured
   Solution: Set appropriate environment variables
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python3 tests/integration/test_llm_intelligence.py
```

## Continuous Integration

### GitHub Actions (Future)

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.12
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python3 -m pytest tests/
```

## Quality Gates

### Before Release
- [ ] All integration tests pass
- [ ] Content enrichment ratios validated
- [ ] LLM analysis quality confirmed
- [ ] OpenDeepResearch integration working
- [ ] CLI commands functional
- [ ] Documentation updated

### Performance Benchmarks
- Content enrichment: <30 seconds per paper
- LLM analysis: <120 seconds per analysis
- Multi-agent coordination: <180 seconds per research topic
- Report generation: <10 seconds per report