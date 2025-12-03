#!/usr/bin/env bash
#
# bench_latency.sh - Run GraphRAG latency benchmarks and collect results
#
# Usage:
#   ./scripts/bench_latency.sh [OPTIONS]
#
# Options:
#   --benchmark TYPE    Benchmark type: all, ttft, e2e, breakdown (default: all)
#   --queries FILE      Custom queries JSON file
#   --output FILE       Output JSON path (default: benchmark_results.json)
#   --compare FILE      Compare with previous benchmark results
#   --no-warmup         Skip cache warmup phase
#   --help              Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
BENCHMARK_TYPE="all"
QUERIES_FILE=""
OUTPUT_FILE="benchmark_results.json"
COMPARE_FILE=""
NO_WARMUP=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark)
            BENCHMARK_TYPE="$2"
            shift 2
            ;;
        --queries)
            QUERIES_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --compare)
            COMPARE_FILE="$2"
            shift 2
            ;;
        --no-warmup)
            NO_WARMUP="--no-warmup"
            shift
            ;;
        --help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GraphRAG Latency Benchmark Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Benchmark type: $BENCHMARK_TYPE"
echo "  Output file: $OUTPUT_FILE"
[[ -n "$QUERIES_FILE" ]] && echo "  Custom queries: $QUERIES_FILE"
[[ -n "$NO_WARMUP" ]] && echo "  Warmup: disabled"
echo ""

# Check if backend is running
echo -e "${YELLOW}Checking backend availability...${NC}"
if ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo -e "${RED}Error: Backend not reachable at http://localhost:8000${NC}"
    echo "Please start the backend with: python api/main.py"
    exit 1
fi
echo -e "${GREEN}✓ Backend is running${NC}"
echo ""

# Build Python command
PYTHON_CMD="python tests/integration_latency.py --benchmark $BENCHMARK_TYPE --output $OUTPUT_FILE"
[[ -n "$QUERIES_FILE" ]] && PYTHON_CMD="$PYTHON_CMD --queries-file $QUERIES_FILE"
[[ -n "$NO_WARMUP" ]] && PYTHON_CMD="$PYTHON_CMD --no-warmup"

# Run benchmark
echo -e "${YELLOW}Running benchmark...${NC}"
echo ""

if eval "$PYTHON_CMD"; then
    echo ""
    echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
    echo -e "Results saved to: ${BLUE}$OUTPUT_FILE${NC}"
else
    echo -e "${RED}✗ Benchmark failed${NC}"
    exit 1
fi

# Compare with previous results if requested
if [[ -n "$COMPARE_FILE" && -f "$COMPARE_FILE" ]]; then
    echo ""
    echo -e "${YELLOW}Comparing with previous results...${NC}"
    python - <<EOF
import json
import sys

def load_results(path):
    with open(path) as f:
        return json.load(f)

def compare_results(current, previous):
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    
    # Compare TTFT
    if "ttft_stats" in current and "ttft_stats" in previous:
        curr = current["ttft_stats"]
        prev = previous["ttft_stats"]
        
        median_diff = curr["median_ms"] - prev["median_ms"]
        median_pct = (median_diff / prev["median_ms"]) * 100
        
        p90_diff = curr["p90_ms"] - prev["p90_ms"]
        p90_pct = (p90_diff / prev["p90_ms"]) * 100
        
        print("\nTime To First Token (TTFT):")
        print(f"  Median: {curr['median_ms']:.1f}ms (was {prev['median_ms']:.1f}ms) {median_diff:+.1f}ms ({median_pct:+.1f}%)")
        print(f"  P90:    {curr['p90_ms']:.1f}ms (was {prev['p90_ms']:.1f}ms) {p90_diff:+.1f}ms ({p90_pct:+.1f}%)")
    
    # Compare E2E cold
    if "e2e_cold_stats" in current and "e2e_cold_stats" in previous:
        curr = current["e2e_cold_stats"]
        prev = previous["e2e_cold_stats"]
        
        median_diff = curr["median_ms"] - prev["median_ms"]
        median_pct = (median_diff / prev["median_ms"]) * 100
        
        p90_diff = curr["p90_ms"] - prev["p90_ms"]
        p90_pct = (p90_diff / prev["p90_ms"]) * 100
        
        print("\nEnd-to-End (Cold):")
        print(f"  Median: {curr['median_ms']:.1f}ms (was {prev['median_ms']:.1f}ms) {median_diff:+.1f}ms ({median_pct:+.1f}%)")
        print(f"  P90:    {curr['p90_ms']:.1f}ms (was {prev['p90_ms']:.1f}ms) {p90_diff:+.1f}ms ({p90_pct:+.1f}%)")
    
    print("\n" + "="*60)

try:
    current = load_results("$OUTPUT_FILE")
    previous = load_results("$COMPARE_FILE")
    compare_results(current, previous)
except Exception as e:
    print(f"Error comparing results: {e}", file=sys.stderr)
    sys.exit(1)
EOF
fi

echo ""
echo -e "${GREEN}Done!${NC}"
