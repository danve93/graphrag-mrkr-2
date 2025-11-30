#!/bin/bash
# Convenient wrapper for GraphRAG startup test
# Usage:
#   ./scripts/test-startup.sh         # Full test suite
#   ./scripts/test-startup.sh quick   # Quick health check
#   ./scripts/test-startup.sh verbose # Full test with verbose logging

cd "$(dirname "$0")/.."

# Use python3 explicitly (more portable)
PYTHON_CMD="${PYTHON:-python3}"

if [ "$1" = "quick" ]; then
    PYTHONPATH=$PWD $PYTHON_CMD scripts/startup_test.py --quick
elif [ "$1" = "verbose" ]; then
    PYTHONPATH=$PWD $PYTHON_CMD scripts/startup_test.py --verbose
else
    PYTHONPATH=$PWD $PYTHON_CMD scripts/startup_test.py
fi
