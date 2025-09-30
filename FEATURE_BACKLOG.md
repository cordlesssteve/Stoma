# Stoma Feature Backlog

**Status**: ACTIVE
**Last Updated**: September 29, 2025

This document tracks planned features, enhancements, and technical debt for the Stoma Research Intelligence System.

## Priority Legend
- 🔴 **Critical**: Blocking issues or essential features
- 🟠 **High**: Important improvements with significant impact
- 🟡 **Medium**: Valuable enhancements, not urgent
- 🟢 **Low**: Nice-to-have features, future considerations

---

## 🔴 Critical Priority

### Documentation Organization (ACTIVE)
**Status**: In Progress
**Priority**: Critical
**Effort**: Medium (2-4 hours)

Move temporary files and reorganize documentation to match CORE standards:
- Populate empty reference categories (05-security, 07-troubleshooting, 08-performance, 09-compliance)
- Move root test files to proper test directories
- Clean up temporary files in root directory
- Document OpenDeepResearch integration architecture

**Dependencies**: None
**Target**: Current session

### Model Selection Optimization
**Status**: Planned
**Priority**: Critical
**Effort**: High (1-2 weeks)

Complete investigation of codellama:13b-instruct parsing issues and establish production model selection guidelines:
- Investigate remaining parsing issues with large models
- Create model selection matrix (reliability vs. quality)
- Implement intelligent fallback system
- Document model-specific quirks and requirements

**Dependencies**: Simplified JSON parser (COMPLETE)
**Target**: Next 2-3 sessions

---

## 🟠 High Priority

### Unit Test Suite Development
**Status**: Planned
**Priority**: High
**Effort**: High (2-3 weeks)

Create comprehensive unit tests for core components:
- Content enrichment components (web scraper, PDF extractor)
- LLM analyzer modules
- Data pipeline components
- Report generation system

**Dependencies**: Test infrastructure setup (pytest.ini, pre-commit)
**Target**: Q4 2025

### Production Pipeline Integration
**Status**: Planned
**Priority**: High
**Effort**: High (2-3 weeks)

Enable LLM analysis in automated production pipelines:
- Integrate simplified parser into main pipeline
- Enable model quality comparison workflows
- Implement automated model validation testing
- Deploy enhanced parsing to report generation

**Dependencies**: Model selection optimization
**Target**: Q4 2025

### Batch Processing System
**Status**: Planned
**Priority**: High
**Effort**: Medium (1 week)

Implement overnight analysis workflows for large-scale processing:
- Queue management for batch operations
- Progress tracking and error handling
- Resource management for local models
- Scheduling integration

**Dependencies**: Production pipeline integration
**Target**: Q4 2025

---

## 🟡 Medium Priority

### Additional Data Sources
**Status**: Planned
**Priority**: Medium
**Effort**: High (varies by source)

Expand collection capabilities:
- Patent database integration (USPTO, EPO)
- News aggregator APIs (NewsAPI, Google News)
- Academic journal APIs (PubMed, JSTOR)
- Podcast/video transcription support

**Dependencies**: Core pipeline stability
**Target**: Q1 2026

### Advanced Analytics Features
**Status**: Planned
**Priority**: Medium
**Effort**: High (3-4 weeks)

Enhance analysis capabilities:
- Predictive trend modeling
- Research impact scoring algorithms
- Technology adoption curve analysis
- Market signal detection
- Competitive intelligence frameworks

**Dependencies**: Sufficient data volume, stable LLM integration
**Target**: Q1 2026

### Report Templates Expansion
**Status**: Planned
**Priority**: Medium
**Effort**: Medium (2-3 weeks)

Create specialized report templates:
- Biotech research landscape reports
- Technology impact assessment templates
- Market analysis frameworks
- Competitive intelligence reports
- Custom report builder UI

**Dependencies**: Advanced analytics features
**Target**: Q1 2026

### Vector Database Integration
**Status**: Planned
**Priority**: Medium
**Effort**: Medium (1-2 weeks)

Add semantic search and similarity analysis:
- ChromaDB or Qdrant integration
- Embedding generation for collected content
- Similarity search across documents
- Topic clustering and visualization

**Dependencies**: Stable content enrichment pipeline
**Target**: Q2 2026

---

## 🟢 Low Priority

### Web Dashboard Enhancement
**Status**: Planned
**Priority**: Low
**Effort**: High (4-6 weeks)

Modernize web interface:
- React/Vue.js frontend rewrite
- Real-time data streaming
- Interactive visualizations
- Customizable dashboards
- Mobile responsive design

**Dependencies**: API stabilization
**Target**: Q2 2026

### Multi-Language Support
**Status**: Planned
**Priority**: Low
**Effort**: High (4-6 weeks)

Add support for non-English sources:
- Multi-language content extraction
- Translation integration
- Language-specific NLP pipelines
- Cross-language similarity analysis

**Dependencies**: Core pipeline stability
**Target**: Q3 2026

### Kubernetes Deployment
**Status**: Planned
**Priority**: Low
**Effort**: High (3-4 weeks)

Production infrastructure for scale:
- Containerization with Docker
- Kubernetes manifests
- Horizontal scaling configuration
- Message queue integration (RabbitMQ/Kafka)
- Monitoring and observability

**Dependencies**: Production validation
**Target**: Q3 2026

---

## 🔧 Technical Debt

### Test Organization
**Status**: Identified
**Priority**: High
**Effort**: Low (1-2 hours)

- Move test files from root to tests/integration/
- Organize tests into logical modules
- Remove redundant/obsolete test files
- Add test documentation

**Target**: Current session

### Code Organization
**Status**: Identified
**Priority**: Medium
**Effort**: Medium (4-8 hours)

- Remove temporary files from root directory
- Consolidate dev scripts into dev-scripts/
- Archive superseded implementation files
- Clean up __pycache__ and temp directories

**Target**: Next session

### Documentation Gaps
**Status**: Identified
**Priority**: Medium
**Effort**: Medium (1 week)

- Add API documentation for all CLI commands
- Document internal module APIs
- Create troubleshooting guides
- Add performance optimization guides
- Security best practices documentation

**Target**: Q4 2025

### Dependency Management
**Status**: Identified
**Priority**: Medium
**Effort**: Low (2-4 hours)

- Audit and remove unused dependencies
- Document optional vs required dependencies
- Create dependency installation guides
- Update requirements.txt with version constraints

**Target**: Q4 2025

---

## 📋 Feature Requests from Users

### Enhanced Report Customization
**Requested**: September 2025
**Status**: Under consideration
**Priority**: TBD

Allow users to define custom report sections, metrics, and formatting preferences through configuration files or UI.

### Citation Graph Visualization
**Requested**: September 2025
**Status**: Under consideration
**Priority**: TBD

Visualize citation networks and paper relationships for better understanding of research landscapes.

---

## 🎯 Completed Features (Archive)

### ✅ OpenDeepResearch Integration (September 29, 2025)
Successfully integrated OpenDeepResearch as git submodule with multi-agent supervisor-researcher architecture and Ollama support.

### ✅ Simplified JSON Parser (September 25, 2025)
Implemented minimal preprocessing approach that resolved parsing issues with larger models while maintaining quality improvements.

### ✅ Ollama Small Model Integration (September 24, 2025)
Production-ready local model integration with report generation and storage capabilities.

### ✅ LLM Analysis Engine (September 2025)
Multi-provider LLM integration with novel contribution detection, research significance scoring, and business intelligence extraction.

### ✅ Content Enrichment Pipeline (September 2025)
48.5x content enhancement through web scraping and PDF extraction, transforming metadata into full-text analysis.

---

## 📝 Notes

### Decision Log
- **2025-09-25**: Removed keyword extraction fallbacks per user feedback - LLM responses already in clean format
- **2025-09-25**: Adopted minimal preprocessing approach for JSON parsing to enable larger models
- **2025-09-24**: Chose Ollama for local model integration over cloud-only approach for cost-free unlimited analysis

### Ideas for Future Consideration
- GraphQL API for flexible data queries
- Webhook system for real-time notifications
- Plugin architecture for custom collectors
- Community marketplace for custom report templates
- Machine learning for automatic source quality assessment

---

**Review Schedule**: Weekly during active development, monthly during maintenance phases
**Stakeholders**: Development team, end users, project maintainers