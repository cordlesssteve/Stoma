# Troubleshooting Guide

This directory contains troubleshooting guides, common issues, and solutions for the Stoma Research Intelligence System.

## Contents

### Planned Documentation

- **Installation Issues** - Common problems during setup and installation
- **Database Connectivity** - PostgreSQL and SQLite connection troubleshooting
- **API Integration Issues** - External API errors and rate limiting
- **LLM Analysis Problems** - Ollama, OpenAI, Anthropic integration issues
- **Content Enrichment** - Web scraping and PDF extraction failures
- **Performance Issues** - Memory usage, slow queries, bottlenecks
- **Testing Problems** - Test failures and debugging strategies

## Common Issues

### Installation & Setup

**Problem**: PostgreSQL connection failures
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql
# Test connection
psql -h localhost -p 5433 -U cordlesssteve -d stoma
```

**Problem**: Missing Python dependencies
```bash
# Reinstall all dependencies
pip install -r requirements.txt
# Or with optional dependencies
pip install -e ".[dev,analysis,vector,web]"
```

### Ollama Integration

**Problem**: Ollama model not found
```bash
# List available models
ollama list
# Pull required model
ollama pull llama3.1:latest
```

**Problem**: JSON parsing errors with LLM responses
- Check model supports instruction format
- Review simplified parser in `minimal_pipeline.py:430-479`
- Try smaller, more reliable models (qwen2.5-coder:3b, gemma2:2b)

### Content Enrichment

**Problem**: PDF extraction failures
- Check Apache Tika installation
- Verify PyMuPDF is installed: `pip install PyMuPDF`
- Try alternative extraction methods (pdfplumber)

**Problem**: Web scraping blocked
- Verify robots.txt compliance
- Check rate limiting configuration
- Update User-Agent string in .env

### Database Issues

**Problem**: PostgreSQL unavailable, system uses SQLite fallback
- Expected behavior - system gracefully degrades
- To re-enable PostgreSQL: Start PostgreSQL service and restart application

**Problem**: Database migration errors
- Backup data before migrations
- Check SQLAlchemy version compatibility
- Review migration scripts for errors

## Emergency Recovery

### If Pipeline Fails
```bash
# Check database connectivity
python3 -c "from stoma.pipeline import DataPipeline; print(DataPipeline().get_pipeline_statistics())"

# Test enrichment components
python3 test_enhanced_pipeline.py

# Verify LLM providers
stoma llm test-providers
```

### If Reports Empty
1. Ensure content enrichment is running (check character count improvements)
2. Verify LLM analysis produces insights (not just metadata)
3. Check pipeline state files in `production_pipeline_data/`

### Reset to Clean State
```bash
# Clear cache and temporary files
rm -rf cache/ temp/ logs/
# Rebuild database (WARNING: deletes all data)
# python3 scripts/rebuild_database.py  # Create this script if needed
```

## Debugging Tips

### Enable Debug Logging
```bash
# In .env file
LOG_LEVEL=DEBUG
```

### Test Individual Components
```bash
# Test ArXiv collection
stoma collect-arxiv -q "test query" -n 1

# Test LLM analysis
stoma llm analyze-text "test text" -o test_output.json

# Test NLP pipeline
stoma nlp batch-analyze --limit 10
```

### Check System Dependencies
```bash
# Python version
python3 --version  # Should be 3.9+

# PostgreSQL version
psql --version

# Ollama status (if using local models)
ollama list
```

## Getting Help

1. **Check logs**: `logs/stoma.log` contains detailed error information
2. **Run diagnostics**: Use built-in test scripts in `tests/integration/`
3. **Review documentation**: Check relevant reference docs for your issue
4. **Search issues**: Look for similar problems in ACTIVE_PLAN.md or FEATURE_BACKLOG.md

---

**Status**: Documentation in progress
**Last Updated**: September 29, 2025