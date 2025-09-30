# Performance Documentation

This directory contains performance optimization guides, benchmarks, and monitoring strategies for the Stoma Research Intelligence System.

## Contents

### Planned Documentation

- **Performance Benchmarks** - Baseline performance metrics and targets
- **Optimization Strategies** - Database queries, caching, batch processing
- **Monitoring Setup** - Metrics collection and alerting
- **Scalability Patterns** - Horizontal scaling and load distribution
- **Resource Management** - Memory usage, CPU optimization, disk I/O
- **Profiling Guide** - Tools and techniques for performance analysis

## Performance Metrics

### Current System Performance

#### Content Enrichment
- **Enhancement Ratio**: 48.5x improvement (2,647 → 128,395 characters)
- **Processing Time**: ~5-10 seconds per ArXiv paper (with PDF extraction)
- **Success Rate**: >95% for PDF extraction

#### LLM Analysis
- **Local Models (Ollama)**:
  - gemma2:2b: ~10-15s per analysis (2.2GB model)
  - llama3.1:8b: ~20-30s per analysis (8GB model)
  - qwen2.5-coder:3b: ~12-18s per analysis (3GB model)
- **Cloud Models**:
  - OpenAI GPT-4: ~3-5s per analysis
  - Anthropic Claude: ~4-6s per analysis

#### Database Operations
- **Write Performance**: 100-200 documents/second (PostgreSQL)
- **Query Performance**: Sub-second for most queries with proper indexing
- **Fallback Mode**: SQLite ~50% slower than PostgreSQL

### Performance Targets

- **Collection Latency**: <5 minutes for new content
- **Analysis Throughput**: 100+ documents/hour (local models)
- **Report Generation**: <30 seconds for standard reports
- **API Response Time**: <2 seconds for most endpoints

## Optimization Strategies

### Database Optimization

#### Indexing Strategy
```sql
-- Key indexes for performance
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_source ON documents(source);
CREATE INDEX idx_documents_status ON documents(status);
```

#### Query Optimization
- Use proper WHERE clause filtering
- Leverage PostgreSQL EXPLAIN ANALYZE
- Implement query result caching
- Use batch inserts for bulk operations

### Caching Strategy

#### Redis Configuration (Optional)
- Cache API responses (TTL: 1 hour)
- Cache enriched content (TTL: 24 hours)
- Cache LLM analysis results (TTL: 7 days)

#### In-Memory Caching
- Use `@lru_cache` for frequently called functions
- Implement request-level caching
- Cache parsed configurations

### LLM Analysis Optimization

#### Model Selection Strategy
1. **Quick Analysis**: Use small models (gemma2:2b, phi3.5:latest)
2. **Standard Analysis**: Use medium models (llama3.1:8b, qwen2.5-coder:3b)
3. **Deep Analysis**: Use large models (llama3.1:70b) or cloud APIs

#### Batch Processing
- Process multiple documents in parallel (max 2-4 concurrent for local models)
- Implement queue-based processing for overnight batch jobs
- Use async/await for I/O-bound operations

### Content Enrichment Optimization

#### Web Scraping
- Respect rate limits (1-10 requests/second)
- Use connection pooling
- Implement exponential backoff for retries
- Cache robots.txt files

#### PDF Extraction
- Try PyMuPDF first (fastest)
- Fallback to pdfplumber if PyMuPDF fails
- Use Apache Tika as last resort (slowest but most reliable)
- Implement timeout for large PDFs (60s default)

## Resource Management

### Memory Management

#### Current Configuration
```python
MAX_WORKERS = 4  # Number of concurrent workers
BATCH_SIZE = 100  # Items per batch
```

#### Memory Guidelines
- Local LLM models require significant RAM:
  - 2B models: 4-6GB RAM
  - 8B models: 12-16GB RAM
  - 70B models: 48-64GB RAM
- Monitor memory usage during batch processing
- Implement memory limits for worker processes

### CPU Optimization
- Use multiprocessing for CPU-intensive tasks
- Implement worker pools for parallel processing
- Profile CPU usage with cProfile
- Consider GPU acceleration for LLM inference (if available)

### Disk I/O
- Use SSD for database storage
- Implement log rotation (max 10MB per file, 5 backups)
- Clean up temporary files regularly
- Monitor disk space usage

## Monitoring & Profiling

### Logging Configuration
```python
LOG_LEVEL=INFO  # Use DEBUG for troubleshooting
LOG_FILE=./logs/stoma.log
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=5
```

### Performance Profiling

#### Python Profiling
```bash
# Profile specific script
python3 -m cProfile -o profile.stats your_script.py
python3 -m pstats profile.stats

# Memory profiling
pip install memory_profiler
python3 -m memory_profiler your_script.py
```

#### Database Profiling
```sql
-- PostgreSQL query analysis
EXPLAIN ANALYZE SELECT * FROM documents WHERE ...;

-- Check slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

## Scalability Considerations

### Horizontal Scaling (Future)
- Message queue integration (RabbitMQ/Kafka)
- Distributed worker pools
- Database replication
- Load balancer for API endpoints

### Current Limitations
- Single-node PostgreSQL (no replication)
- Local LLM inference (no GPU cluster)
- Synchronous processing for some operations

---

**Status**: Documentation in progress
**Last Updated**: September 29, 2025