#!/bin/bash

# Graph quick start script (branding pulled from `branding.json`)

set -e

# Read branding title from branding.json (falls back to `title` if `setup_title` missing)
BRANDING_TITLE=$(python3 - <<'PY'
import json
import sys
try:
    b = json.load(open('frontend/public/branding.json'))
    print(b.get('setup_title') or b.get('title'))
except Exception:
    print('Amber is waking up')
PY
)

echo "$BRANDING_TITLE"
echo "====================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validate Python and Node.js versions
PYTHON_VERSION=$(python3 - <<'EOF'
import sys
print('.'.join(map(str, sys.version_info[:3])))
EOF
)

NODE_VERSION=$(node -v 2>/dev/null | sed 's/^v//')

if [ -z "$NODE_VERSION" ]; then
    echo "${YELLOW}Node.js is not installed. Please install Node.js 18 or higher before continuing.${NC}"
    exit 1
fi

version_ge() {
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

if ! version_ge "$PYTHON_VERSION" "3.10"; then
    echo "${YELLOW}Python 3.10 or higher is required. Detected: ${PYTHON_VERSION}.${NC}"
    exit 1
fi

if ! version_ge "$NODE_VERSION" "18.0.0"; then
    echo "${YELLOW}Node.js 18 or higher is required. Detected: ${NODE_VERSION}.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install Python dependencies
echo "${GREEN}Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "${YELLOW}Creating .env file from example...${NC}"
    cat > .env << 'EOF'
# Environment Configuration Template
# Copy this file to .env and update with your actual values

# LLM / Ollama Configuration
# Set provider to 'ollama' to use a local Ollama server for embeddings and LLM
# Supported values: 'openai' (default) or 'ollama'
LLM_PROVIDER=openai

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1/
OPENAI_MODEL=gpt-4o-mini
OPENAI_PROXY=

# Neo4j Configuration
# Use bolt://neo4j:7687 for Docker Compose, bolt://localhost:7687 for local Python scripts
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
# Optional Docker Compose credential shortcut: set NEO4J_AUTH to '<user>/<password>'
# If set, docker-compose will use NEO4J_AUTH (e.g. NEO4J_AUTH=neo4j/test). When running
# the backend/service locally outside Compose, keep NEO4J_USERNAME/NEO4J_PASSWORD set for the
# application to read.
# Example for local demo (safe default used by compose in this repo):
NEO4J_AUTH=neo4j/test

# If using Ollama, set the base URL (e.g. http://host.docker.internal:11434 or http://ollama:11434)
#OLLAMA_BASE_URL=http://localhost:11434
#OLLAMA_MODEL=llama3.2
#OLLAMA_EMBEDDING_MODEL=nomic-embed-text
#OLLAMA_API_KEY=   # Not required for local Ollama server

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_CONCURRENCY=3

# Document Processing Configuration (Optimized for Company Docs)
CHUNK_SIZE=1200
CHUNK_OVERLAP=150

# Application Configuration
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=104857600

# Feature Flags (Optimized for Quality)
ENABLE_ENTITY_EXTRACTION=true
ENABLE_QUALITY_SCORING=true
ENABLE_DELETE_OPERATIONS=true
FLASHRANK_ENABLED=true
FLASHRANK_MODEL_NAME=ms-marco-TinyBERT-L-2-v2

# Microsoft GraphRAG Phases (Enabled for Quality)
ENABLE_GLEANING=true
MAX_GLEANINGS=1
ENABLE_PHASE2_NETWORKX=true
ENABLE_DESCRIPTION_SUMMARIZATION=true

# Marker PDF Conversion (Highest Accuracy)
USE_MARKER_FOR_PDF=true
MARKER_USE_LLM=true
MARKER_FORCE_OCR=true
MARKER_OUTPUT_FORMAT=markdown
MARKER_LLM_MODEL=gpt-4o-mini
# SECURITY: API keys must ONLY be set here in .env, never in JSON config or code
# Optional: Separate API key for Marker (defaults to OPENAI_API_KEY if not set)
# MARKER_LLM_API_KEY=your_openai_api_key_here
# Optional: Set MARKER_LLM_SERVICE to use different LLM service class
# MARKER_LLM_SERVICE=marker.services.openai.OpenAIService

# Caching Configuration
ENTITY_LABEL_CACHE_SIZE=5000
ENTITY_LABEL_CACHE_TTL=300
EMBEDDING_CACHE_SIZE=10000
RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60
NEO4J_MAX_CONNECTION_POOL_SIZE=50
ENABLE_CACHING=true

# Additional Marker options (advanced)
MARKER_PAGINATE_OUTPUT=true
MARKER_STRIP_EXISTING_OCR=false
MARKER_PDFTEXT_WORKERS=4

# Redis Configuration (optional, for background job processing)
# REDIS_URL=redis://localhost:6379/0
EOF
    echo "${YELLOW}⚠️  Please edit .env and add your API keys${NC}"
fi

# Setup frontend
echo "${GREEN}Setting up frontend...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    echo "${GREEN}Installing Node.js dependencies...${NC}"
    npm install
fi

if [ ! -f ".env.local" ]; then
    echo "${GREEN}Creating frontend .env.local...${NC}"
    # Default to local backend for non-Docker runs; Docker Compose overrides this to http://backend:8000
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
fi

cd ..

echo ""
echo "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key and Neo4j credentials"
echo "2. Make sure Neo4j is running"
echo "3. Start the backend: python api/main.py"
echo "4. In another terminal, start the frontend: cd frontend && npm run dev"
echo "5. Open http://localhost:3000 in your browser"
echo ""
echo "For detailed instructions, see README.md and AGENTS.md"

# Docker Compose alternative
echo "\nAlternate Docker-based startup (recommended for demos):"
echo "1. Make sure Docker and Docker Compose are installed and running"
echo "2. From the project root run:\n   docker compose up -d\n"
echo "3. To rebuild images after changing Dockerfiles, run:\n   docker compose up -d --build\n"

echo "If you prefer the containerized path, skip starting the backend/frontend locally and use the docker-compose stack instead."
