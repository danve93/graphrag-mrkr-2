#!/bin/bash

# Graph quick start script (branding pulled from `branding.json`)

set -e

# Read branding title from branding.json (falls back to `title` if `setup_title` missing)
BRANDING_TITLE=$(python3 - <<'PY'
import json
import sys
try:
    b = json.load(open('branding.json'))
    print(b.get('setup_title') or b.get('title'))
except Exception:
    print('GraphRAG v2.0 Setup')
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
    cat > .env << EOF
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4jpassword

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002

# Application Configuration
LOG_LEVEL=INFO
ENABLE_CLUSTERING=true
ENABLE_GRAPH_CLUSTERING=true
SUMMARIZATION_BATCH_SIZE=20
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
echo "For detailed instructions, see SETUP_V2.md"

# Docker Compose alternative
echo "\nAlternate Docker-based startup (recommended for demos):"
echo "1. Make sure Docker and Docker Compose (v2) are installed and running"
echo "2. From the project root run:\n   docker compose up -d\n"
echo "3. To rebuild images after changing Dockerfiles, run:\n   docker compose up -d --build\n"

echo "If you prefer the containerized path, skip starting the backend/frontend locally and use the docker-compose stack instead."
