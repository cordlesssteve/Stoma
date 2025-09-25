# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KnowHunt is a Research Intelligence System that automates collection, analysis, and reporting of information from academic papers, public documents, corporate intelligence, and software projects. The system features a complete content enrichment pipeline that transforms basic metadata collection into comprehensive intelligence analysis.

## Development Commands

### Core Testing & Development
```bash
# Test enhanced pipeline with content enrichment
python3 test_enhanced_pipeline.py

# Test real pipeline with ArXiv papers
python3 test_real_pipeline.py

# Test LLM intelligence integration
python3 test_llm_intelligence.py

# Test Ollama local model integration
python3 test_ollama_integration.py

# Run specific test file
python3 test_nlp_pipeline.py
```

### CLI Usage
```bash
# Main CLI entry point
knowhunt --help

# Collect and analyze ArXiv papers
knowhunt collect-arxiv -q "machine learning" -n 5 -o results.json

# LLM analysis commands
knowhunt llm analyze-text "research text here" -o analysis.json
knowhunt llm collect-and-analyze-arxiv -q "protein folding" -n 3
knowhunt llm test-providers

# Report management commands (NEW)
knowhunt llm search-reports --query "protein folding" --min-quality 5
knowhunt llm view-report cli_analysis_20250924_152955
knowhunt llm storage-stats

# NLP analysis commands
knowhunt nlp batch-analyze --limit 100
knowhunt nlp detect-trends --timeframe 30
```

### Setup & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Optional development dependencies
pip install -e ".[dev,analysis,vector,web]"

# Setup local environment
./setup_local.sh

# Setup lightweight LLM (Ollama)
./setup_lightweight_llm.sh
```

## Architecture Overview

### Content Enhancement Pipeline
The system's core strength is its **48.5x content enhancement** capability:

- **Collection**: ArXiv, Reddit, HackerNews, SEC filings, GitHub
- **Enrichment**: Web scraping + PDF extraction transforms metadata into full content
- **Analysis**: Traditional NLP + LLM intelligence for genuine insights
- **Reporting**: Data-driven reports with actionable intelligence

### Key Components

#### 1. Enhanced Data Pipeline (`knowhunt/pipeline/`)
- `data_pipeline.py` - Complete collection→enrichment→analysis→reporting flow
- `data_types.py` - Structured data types for pipeline operations
- **Critical**: Always use enrichment cycles for meaningful analysis

#### 2. Content Enrichment System (`knowhunt/enrichment/`)
- `web_scraper.py` - Respectful web scraping with robots.txt compliance
- `pdf_extractor.py` - Multi-method PDF extraction (PyMuPDF, pdfplumber, Tika)
- `content_enricher.py` - Orchestrates enhancement strategies
- **Achievement**: Transforms 72-character headlines into full articles

#### 3. LLM Analysis Engine (`knowhunt/analysis/llm_analyzer.py`)
- **NEW**: Production-ready intelligent analysis system
- **Providers**: OpenAI, Anthropic, Ollama (local models)
- **Capabilities**: Novel contribution detection, research significance scoring, business intelligence
- **Models**: Supports cloud (GPT-4, Claude) and local (Llama, Gemma) models

#### 4. Traditional NLP Pipeline (`knowhunt/analysis/`)
- `nlp_analyzer.py` - SpaCy + NLTK + TextBlob processing
- `trend_detector.py` - Keyword trends and emerging topics
- `correlation_analyzer.py` - Cross-paper relationship analysis
- **Note**: Maintained alongside LLM analysis, not replaced by it

#### 5. Data-Driven Reports (`knowhunt/reports/`)
- `data_driven_generator.py` - Real intelligence reports (not templates)
- **Critical**: Reports now contain actionable insights vs. metadata summaries
- **Success**: Resolved "essentially useless" user feedback

#### 6. Report Storage & Management (`knowhunt/storage/`)
- **NEW**: `report_manager.py` - Organized storage with automatic indexing
- **NEW**: `database.py` - PostgreSQL integration with graceful SQLite fallback
- **Features**: Dual storage (PostgreSQL + SQLite), automatic organization by provider/date
- **CLI Integration**: Search, view, and manage analysis reports through CLI commands

### Collection Sources
- **ArXiv**: Academic papers with PDF extraction
- **Reddit**: Community discussions with enhanced content
- **HackerNews**: Technology trends with full article content
- **SEC EDGAR**: Corporate filings with document parsing
- **GitHub**: Repository analysis with README extraction

## Development Standards

### Content Quality Focus
- **Always prioritize** full content over metadata
- **Maintain** 48.5x+ enhancement ratios achieved in testing
- **Use enrichment pipeline** for all meaningful analysis

### Respectful Data Collection
- **Mandatory**: robots.txt compliance for web scraping
- **Required**: Rate limiting (1-10 requests/second based on source)
- **Ethics**: Sustainable, long-term collection practices

### LLM Integration Guidelines
- **Local First**: Prefer Ollama for cost-free unlimited analysis
- **Cloud Fallback**: Use OpenAI/Anthropic for specialized tasks
- **Model Selection**: Match model size to content complexity
  - `gemma2:2b` (2.2GB) - Quick analysis, testing
  - `llama3.1:8b` (8GB) - Standard papers
  - `llama3.1:70b` (40GB) - Complex research, production

### Database Configuration
- **PostgreSQL**: Primary storage with graceful fallbacks (requires psycopg2)
- **Connection**: Local peer authentication on port 5433
- **SQLite**: Automatic fallback for report indexing when PostgreSQL unavailable
- **Dual Storage**: Files stored on disk, metadata indexed in database

### Testing Requirements
- **Always verify** enhancement ratios before claiming success
- **Test enrichment** with real ArXiv papers, not mock data
- **Measure content growth**: Track character count improvements
- **Validate intelligence**: Ensure LLM outputs contain genuine insights

## Project Status (September 2025)

### ✅ Complete & Operational
1. **Content Enrichment Pipeline**: 48.5x improvement demonstrated
2. **LLM Intelligence Integration**: Multi-provider analysis system
3. **Data-Driven Reporting**: Transformed from "useless" to valuable
4. **Ollama Local Models**: Cost-free unlimited analysis capability
5. **CLI Interface**: Production-ready commands for all workflows
6. **Report Storage & Management**: Organized storage with PostgreSQL/SQLite indexing
7. **Search & Discovery**: Full-text search and report management through CLI

### 🎯 Current Focus Areas
1. **Production Deployment**: Enable LLM analysis in automated pipelines
2. **Batch Processing**: Overnight analysis workflows
3. **Model Optimization**: Fine-tune model selection for different content types
4. **Performance Scaling**: Handle larger content volumes efficiently

## Critical Success Factors

### User Feedback Resolution
- **Previous Problem**: "The report is essentially useless, from a conceptual point of view"
- **Solution Delivered**: LLM-powered analysis with novel contribution detection
- **Achievement**: Complete transformation from metadata to intelligence

### Technical Breakthroughs
- **Content Enhancement**: 48.5x improvement (2,647 → 128,395 characters)
- **Intelligence Analysis**: Genuine research insights vs. keyword fragments
- **Local Model Support**: Zero API costs with Ollama integration

### Development Priorities
1. **Content Quality**: Always full content over metadata summaries
2. **Respectful Practices**: robots.txt compliance and rate limiting
3. **Verification**: Test all claims with real data, not theoretical capabilities
4. **User Value**: Focus on actionable intelligence vs. technical metrics

## Emergency Recovery

### If Pipeline Fails
1. Check database connectivity with `python3 -c "from knowhunt.pipeline import DataPipeline; print(DataPipeline().get_pipeline_statistics())"`
2. Test enrichment components with `python3 test_enhanced_pipeline.py`
3. Verify LLM providers with `knowhunt llm test-providers`

### If Reports Empty
1. Ensure content enrichment is running (check character count improvements)
2. Verify LLM analysis produces insights (not just metadata)
3. Check pipeline state files in `production_pipeline_data/`

This system represents a complete transformation from basic data collection to sophisticated research intelligence with genuine analytical value.