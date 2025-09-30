# Quick Integration Guide - High-Value Tools

## 🚀 Immediate Wins (Can implement in hours)

### 1. Apache Tika - Universal Document Parser
**Why**: Handles 1000+ file formats, including PDFs, Word, Excel
**Time to integrate**: 2 hours

```bash
# Start Tika server
docker run -d -p 9998:9998 apache/tika
```

```python
# stoma/parsers/tika_parser.py
import requests

class TikaParser:
    def __init__(self, tika_url="http://localhost:9998"):
        self.tika_url = tika_url
    
    def parse_document(self, file_path):
        """Extract text and metadata from any document"""
        with open(file_path, 'rb') as f:
            response = requests.put(
                f"{self.tika_url}/tika",
                data=f,
                headers={"Accept": "text/plain"}
            )
        return response.text

    def get_metadata(self, file_path):
        """Extract just metadata"""
        with open(file_path, 'rb') as f:
            response = requests.put(
                f"{self.tika_url}/meta",
                data=f,
                headers={"Accept": "application/json"}
            )
        return response.json()
```

### 2. Metabase - Instant Analytics Dashboard
**Why**: Beautiful dashboards without coding
**Time to integrate**: 1 hour

```bash
# Start Metabase
docker run -d -p 3001:3000 \
  -e "MB_DB_TYPE=postgres" \
  -e "MB_DB_DBNAME=stoma" \
  -e "MB_DB_PORT=5432" \
  -e "MB_DB_USER=stoma" \
  -e "MB_DB_PASS=stoma" \
  -e "MB_DB_HOST=host.docker.internal" \
  --name metabase metabase/metabase
```

Then visit http://localhost:3001 and connect to your PostgreSQL database.
Pre-built dashboards for:
- Collection statistics
- Trend analysis
- Source performance
- Search patterns

### 3. Meilisearch - Better Search in 10 Minutes
**Why**: Typo-tolerant, instant search, easier than Elasticsearch
**Time to integrate**: 30 minutes

```bash
# Start Meilisearch
docker run -d -p 7700:7700 \
  -v $(pwd)/meili_data:/meili_data \
  getmeili/meilisearch:latest
```

```python
# stoma/search/meilisearch_client.py
import meilisearch

class BetterSearch:
    def __init__(self):
        self.client = meilisearch.Client('http://localhost:7700')
        self.index = self.client.index('research')
    
    def index_document(self, doc):
        """Add document to search index"""
        self.index.add_documents([{
            'id': doc['id'],
            'title': doc['title'],
            'content': doc['content'],
            'source': doc['source_type'],
            'date': doc['published_date']
        }])
    
    def search(self, query):
        """Search with typo tolerance and ranking"""
        return self.index.search(query, {
            'limit': 20,
            'attributesToHighlight': ['title', 'content']
        })
```

### 4. MinIO - S3-Compatible Storage
**Why**: Store PDFs and large files cheaply
**Time to integrate**: 1 hour

```bash
# Start MinIO
docker run -d -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=stoma" \
  -e "MINIO_ROOT_PASSWORD=stoma123" \
  -v $(pwd)/minio_data:/data \
  minio/minio server /data --console-address ":9001"
```

```python
# stoma/storage/object_storage.py
from minio import Minio
import io

class ObjectStorage:
    def __init__(self):
        self.client = Minio(
            "localhost:9000",
            access_key="stoma",
            secret_key="stoma123",
            secure=False
        )
        self.bucket = "stoma-docs"
        
        # Create bucket if not exists
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
    
    def store_file(self, file_id, file_content, content_type="application/pdf"):
        """Store file in object storage"""
        self.client.put_object(
            self.bucket,
            file_id,
            io.BytesIO(file_content),
            len(file_content),
            content_type=content_type
        )
    
    def get_file(self, file_id):
        """Retrieve file from storage"""
        response = self.client.get_object(self.bucket, file_id)
        return response.read()
```

## 📊 Next Level Integrations (1-2 days each)

### 5. Streamlit - Quick Data Apps
**Why**: Python-native, no frontend skills needed
**Example**: Research paper explorer

```python
# apps/paper_explorer.py
import streamlit as st
import pandas as pd
from stoma.storage.base import PostgreSQLStorage

st.title("📚 Stoma Paper Explorer")

# Search box
query = st.text_input("Search papers", "machine learning")

if st.button("Search"):
    storage = PostgreSQLStorage(config)
    results = storage.search(query)
    
    df = pd.DataFrame(results)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Results", len(df))
    col2.metric("Sources", df['source_type'].nunique())
    col3.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")
    
    # Show results
    for _, row in df.iterrows():
        with st.expander(row['title']):
            st.write(row['summary'])
            st.write(f"Source: {row['source_type']}")
            st.write(f"Date: {row['published_date']}")
```

Run with: `streamlit run apps/paper_explorer.py`

### 6. Prefect - Better Scheduler
**Why**: More robust than our scheduler, better monitoring
**Migration path**: Keep both, migrate gradually

```python
# flows/daily_collection.py
from prefect import flow, task
from stoma.collectors.arxiv import ArXivCollector

@task(retries=3)
def collect_arxiv():
    collector = ArXivCollector(config)
    results = []
    for item in collector.collect():
        results.append(item)
    return results

@task
def store_results(results):
    storage = PostgreSQLStorage(config)
    for result in results:
        storage.store(result)

@flow(name="Daily Research Collection")
def daily_collection():
    arxiv_results = collect_arxiv()
    store_results(arxiv_results)

# Deploy and schedule
if __name__ == "__main__":
    daily_collection.serve(
        name="stoma-daily",
        cron="0 9 * * *"  # 9 AM daily
    )
```

### 7. Sentence Transformers - Semantic Search
**Why**: Find similar papers, not just keyword matches
**Use case**: "Papers similar to this one"

```python
# stoma/analysis/semantic_search.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
    
    def build_index(self, documents):
        """Build semantic search index"""
        self.documents = documents
        
        # Generate embeddings
        texts = [f"{d['title']} {d['summary']}" for d in documents]
        embeddings = self.model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def find_similar(self, query, k=10):
        """Find semantically similar documents"""
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return similar documents
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(1 / (1 + dist))  # Convert distance to similarity
            results.append(doc)
        
        return results
```

## 🎯 Docker Compose for All Services

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: stoma
      POSTGRES_USER: stoma
      POSTGRES_PASSWORD: stoma
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  tika:
    image: apache/tika
    ports:
      - "9998:9998"

  meilisearch:
    image: getmeili/meilisearch:latest
    ports:
      - "7700:7700"
    volumes:
      - meilisearch_data:/meili_data

  metabase:
    image: metabase/metabase
    ports:
      - "3001:3000"
    environment:
      MB_DB_TYPE: postgres
      MB_DB_DBNAME: stoma
      MB_DB_PORT: 5432
      MB_DB_USER: stoma
      MB_DB_PASS: stoma
      MB_DB_HOST: postgres
    depends_on:
      - postgres

  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    environment:
      MINIO_ROOT_USER: stoma
      MINIO_ROOT_PASSWORD: stoma123
    command: server /data --console-address ":9001"

volumes:
  postgres_data:
  meilisearch_data:
  minio_data:
```

Start everything: `docker-compose up -d`

## 📈 Impact Assessment

| Tool | Setup Time | Impact | Priority |
|------|-----------|--------|----------|
| Apache Tika | 2 hours | Handles all document types | **Critical** |
| Metabase | 1 hour | Instant analytics | **High** |
| Meilisearch | 30 min | 10x better search | **High** |
| MinIO | 1 hour | Scalable file storage | **Medium** |
| Streamlit | 2 hours | Quick custom dashboards | **Medium** |
| Prefect | 4 hours | Production scheduler | **Low** (we have one) |
| Semantic Search | 2 hours | Similar paper finding | **Medium** |

## 🏃 Quick Start Commands

```bash
# 1. Start all services
docker-compose up -d

# 2. Install Python libraries
pip install python-tika meilisearch-python minio sentence-transformers streamlit

# 3. Test Tika
curl -T test.pdf http://localhost:9998/tika --header "Accept: text/plain"

# 4. Access services
# Metabase: http://localhost:3001
# MinIO Console: http://localhost:9001
# Meilisearch: http://localhost:7700
```

## 💡 Pro Tips

1. **Start with Tika + Metabase** - Biggest bang for buck
2. **Use Meilisearch over Elasticsearch** - Simpler, good enough
3. **Keep PostgreSQL as primary** - These tools complement, not replace
4. **Gradual migration** - Don't rip out working code
5. **Document everything** - These add complexity

---

*These integrations can transform Stoma from prototype to production-ready in days, not months.*