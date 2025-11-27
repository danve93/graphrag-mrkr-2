#!/bin/bash

# Script to run the chat pipeline test with Docker Compose
# This ensures Neo4j and other services are available

set -e

echo "=================================================="
echo "Chat Pipeline Test Runner"
echo "=================================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please copy .env.example to .env and configure it"
    exit 1
fi

# Source .env file to get credentials
set -a
source .env
set +a

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check if Neo4j is already running and healthy
if docker ps | grep -q neo4j; then
    echo "✓ Neo4j container is already running"
    
    # Quick health check
    if docker exec neo4j cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" "RETURN 1" > /dev/null 2>&1; then
        echo "✓ Neo4j is healthy and accepting connections"
    else
        echo "⚠ Neo4j is running but not ready yet, waiting..."
        MAX_ATTEMPTS=15
        ATTEMPT=0
        
        while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
            if docker exec neo4j cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" "RETURN 1" > /dev/null 2>&1; then
                echo "✓ Neo4j is now ready"
                break
            fi
            ATTEMPT=$((ATTEMPT + 1))
            echo "  Waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
            sleep 2
        done
        
        if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
            echo "❌ Error: Neo4j failed to become ready within timeout"
            echo "Check logs: docker-compose logs neo4j"
            exit 1
        fi
    fi
else
    # Start Neo4j if not running
    echo "Starting Neo4j..."
    docker-compose up -d neo4j
    
    # Wait for Neo4j to be ready
    echo "Waiting for Neo4j to be ready..."
    MAX_ATTEMPTS=30
    ATTEMPT=0
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if docker exec neo4j cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" "RETURN 1" > /dev/null 2>&1; then
            echo "✓ Neo4j is ready"
            break
        fi
        ATTEMPT=$((ATTEMPT + 1))
        echo "  Waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
        sleep 2
    done
    
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        echo "❌ Error: Neo4j failed to start within timeout"
        echo "Check logs: docker-compose logs neo4j"
        exit 1
    fi
fi

echo ""
echo "=================================================="
echo "Running Chat Pipeline Tests"
echo "=================================================="
echo ""

# Update NEO4J_URI to use localhost for tests running outside Docker
export NEO4J_URI="bolt://localhost:7687"

# Run the test (use python3 explicitly)
python3 -m pytest api/tests/test_chat_pipeline.py -v -s "$@"
TEST_EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Test Results"
echo "=================================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "❌ Some tests failed (exit code: $TEST_EXIT_CODE)"
fi

echo ""
echo "To stop Neo4j, run: docker-compose down"
echo ""

exit $TEST_EXIT_CODE
