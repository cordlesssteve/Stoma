#!/bin/bash
# Local Development Setup Script for KnowHunt

echo "🔍 Setting up KnowHunt for Local Development"
echo "================================================"

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{pdfs,cache,exports,logs}
echo "✓ Data directories created"

# Check if PostgreSQL is installed and running
echo "🗄️ Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "✓ PostgreSQL found"
    
    # Check if knowhunt database exists
    if psql -lqt | cut -d \| -f 1 | grep -qw knowhunt; then
        echo "✓ knowhunt database already exists"
    else
        echo "Creating knowhunt database..."
        createdb knowhunt
        echo "✓ knowhunt database created"
    fi
else
    echo "❌ PostgreSQL not found. Please install it first:"
    echo "   Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "   macOS: brew install postgresql"
    echo "   Then start the service and run this script again."
    exit 1
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -e .
echo "✓ Python dependencies installed"

# Test database connection
echo "🔌 Testing database connection..."
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from knowhunt.storage.base import PostgreSQLStorage
from knowhunt.config.settings import load_config

async def test_db():
    try:
        config = load_config('config_local.yaml')
        storage = PostgreSQLStorage(config['storage'])
        await storage.connect()
        await storage.disconnect()
        print('✓ Database connection successful')
        return True
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        return False

success = asyncio.run(test_db())
sys.exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    echo "✓ Database connection test passed"
else
    echo "❌ Database connection test failed"
    echo "   Try: sudo -u postgres createuser -s $USER"
    echo "   Then run this script again"
    exit 1
fi

# Initialize database schema
echo "🏗️ Initializing database schema..."
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from knowhunt.storage.base import PostgreSQLStorage
from knowhunt.config.settings import load_config

async def init_schema():
    try:
        config = load_config('config_local.yaml')
        storage = PostgreSQLStorage(config['storage'])
        await storage.connect()
        await storage.ensure_tables()
        await storage.disconnect()
        print('✓ Database schema initialized')
    except Exception as e:
        print(f'❌ Schema initialization failed: {e}')

asyncio.run(init_schema())
"

# Test basic collection
echo "🧪 Testing basic collection..."
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from knowhunt.collectors.arxiv import ArXivCollector

async def test_collection():
    try:
        collector = ArXivCollector({'max_results': 3, 'rate_limit': 1.0})
        results = []
        async for result in collector.collect('machine learning'):
            results.append(result)
            if len(results) >= 2:  # Just test with 2 papers
                break
        print(f'✓ Collection test passed - collected {len(results)} papers')
        if results:
            print(f'  Sample: {results[0].data.get(\"title\", \"Unknown\")[:50]}...')
    except Exception as e:
        print(f'❌ Collection test failed: {e}')

asyncio.run(test_collection())
"

echo ""
echo "🎉 Local setup complete!"
echo ""
echo "Quick start commands:"
echo "  Test collectors:     python3 test_pipeline.py"
echo "  Start web dashboard: python3 -m knowhunt.api.main"
echo "  Run CLI commands:    python3 -m knowhunt.cli.main --help"
echo ""
echo "Configuration file: config_local.yaml"
echo "Data directory:     ./data/"
echo ""
echo "Next steps:"
echo "1. python3 -m knowhunt.cli.main collect-arxiv -q 'machine learning' -n 5"
echo "2. python3 -m knowhunt.api.main (then visit http://localhost:8000)"
echo "3. python3 -m knowhunt.scheduler.manager (for automated collection)"