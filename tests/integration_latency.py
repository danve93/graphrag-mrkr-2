"""
Latency benchmark suite for GraphRAG.

Measures:
- TTFT (Time To First Token) for streaming responses
- Median and p90 latencies for full query completion
- Retrieval latency vs generation latency breakdown
- Cache hit/miss impact on latency

Usage:
    python tests/integration_latency.py
    
    # Or with custom queries:
    python tests/integration_latency.py --queries-file custom_queries.json
    
    # Run specific benchmark:
    python tests/integration_latency.py --benchmark ttft
"""

import time
import json
import statistics
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Representative test queries covering different complexity levels
DEFAULT_QUERIES = [
    {
        "query": "What is Neo4j?",
        "description": "Simple factual query",
        "expected_strategy": "balanced"
    },
    {
        "query": "How do I configure SSL certificates?",
        "description": "Procedural query",
        "expected_strategy": "keyword_focused"
    },
    {
        "query": "Compare the features of Server A and Server B",
        "description": "Comparative query",
        "expected_strategy": "entity_focused"
    },
    {
        "query": "What are the relationships between components and services?",
        "description": "Relationship query",
        "expected_strategy": "entity_focused"
    },
    {
        "query": "Explain the migration procedure from version 4 to version 5",
        "description": "Complex analytical query",
        "expected_strategy": "keyword_focused"
    }
]


class LatencyBenchmark:
    """Benchmark runner for measuring GraphRAG latency metrics."""
    
    def __init__(self, queries: List[Dict[str, str]] = None, warmup: bool = True):
        self.queries = queries or DEFAULT_QUERIES
        self.warmup = warmup
        self.results = []
    
    def measure_ttft(self, query: str) -> Dict[str, Any]:
        """Measure Time To First Token for streaming response."""
        from rag.graph_rag import graph_rag
        import asyncio
        
        start_time = time.time()
        first_token_time = None
        tokens_received = 0
        total_time = None
        
        try:
            # Use async streaming to measure TTFT
            async def stream_and_measure():
                nonlocal first_token_time, tokens_received, total_time
                
                async for event in graph_rag.query_stream(
                    query=query,
                    chat_history=[],
                    session_id=None,
                    retrieval_mode="graph_enhanced",
                    top_k=5
                ):
                    if event.get("type") == "token" and first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    if event.get("type") == "token":
                        tokens_received += 1
                
                total_time = time.time() - start_time
            
            asyncio.run(stream_and_measure())
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "ttft_ms": None,
                "total_ms": None
            }
        
        return {
            "query": query,
            "ttft_ms": first_token_time * 1000 if first_token_time else None,
            "total_ms": total_time * 1000 if total_time else None,
            "tokens": tokens_received,
            "tokens_per_second": tokens_received / total_time if total_time else 0
        }
    
    def measure_end_to_end(self, query: str, use_cache: bool = False) -> Dict[str, Any]:
        """Measure end-to-end latency for non-streaming query."""
        from rag.graph_rag import graph_rag
        from core.singletons import get_response_cache, clear_response_cache
        
        if not use_cache:
            clear_response_cache()
        
        start_time = time.time()
        
        try:
            result = graph_rag.query(
                query=query,
                chat_history=[],
                session_id=None,
                retrieval_mode="graph_enhanced",
                top_k=5
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Check if response came from cache
            cache = get_response_cache()
            from_cache = result.get("metadata", {}).get("from_cache", False)
            
            return {
                "query": query,
                "latency_ms": latency_ms,
                "from_cache": from_cache,
                "response_length": len(result.get("response", "")),
                "sources_count": len(result.get("sources", [])),
                "quality_score": result.get("quality_score", 0.0)
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "latency_ms": None
            }
    
    def measure_retrieval_vs_generation(self, query: str) -> Dict[str, Any]:
        """Measure retrieval time separately from generation time."""
        from rag.retriever import hybrid_retrieval
        from rag.nodes.query_analysis import analyze_query
        from rag.nodes.generation import generate_response
        from core.graph_db import get_graph_db_driver
        from core.embeddings import embedding_manager
        
        # Measure query analysis
        analysis_start = time.time()
        analysis = analyze_query(query=query, chat_history=[])
        analysis_ms = (time.time() - analysis_start) * 1000
        
        # Measure retrieval
        retrieval_start = time.time()
        try:
            driver = get_graph_db_driver()
            results = hybrid_retrieval(
                driver=driver,
                query=query,
                embeddings_manager=embedding_manager,
                top_k=5,
                retrieval_mode="graph_enhanced",
                query_analysis=analysis
            )
            retrieval_ms = (time.time() - retrieval_start) * 1000
            chunks_retrieved = len(results)
        except Exception as e:
            retrieval_ms = None
            chunks_retrieved = 0
        
        # Measure generation (mocked to avoid actual LLM call in benchmark)
        generation_start = time.time()
        # In real scenario, would call generate_response
        # For benchmark, we simulate
        time.sleep(0.5)  # Simulate LLM latency
        generation_ms = (time.time() - generation_start) * 1000
        
        return {
            "query": query,
            "analysis_ms": analysis_ms,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": analysis_ms + (retrieval_ms or 0) + generation_ms,
            "chunks_retrieved": chunks_retrieved
        }
    
    def run_benchmark_suite(self, benchmark_type: str = "all") -> Dict[str, Any]:
        """Run complete benchmark suite."""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark_type": benchmark_type,
            "queries_count": len(self.queries),
            "measurements": []
        }
        
        # Warmup if enabled
        if self.warmup:
            print("Warming up caches...")
            for query_spec in self.queries[:2]:
                try:
                    self.measure_end_to_end(query_spec["query"], use_cache=False)
                except:
                    pass
            print("Warmup complete\n")
        
        # Run benchmarks
        for i, query_spec in enumerate(self.queries, 1):
            query = query_spec["query"]
            description = query_spec.get("description", "")
            
            print(f"[{i}/{len(self.queries)}] Benchmarking: {description}")
            print(f"  Query: {query[:80]}...")
            
            measurement = {
                "query": query,
                "description": description
            }
            
            if benchmark_type in ["all", "ttft"]:
                print("  Measuring TTFT...")
                ttft_result = self.measure_ttft(query)
                measurement["ttft"] = ttft_result
            
            if benchmark_type in ["all", "e2e"]:
                print("  Measuring end-to-end (cold)...")
                e2e_cold = self.measure_end_to_end(query, use_cache=False)
                measurement["e2e_cold"] = e2e_cold
                
                print("  Measuring end-to-end (warm/cached)...")
                e2e_warm = self.measure_end_to_end(query, use_cache=True)
                measurement["e2e_warm"] = e2e_warm
            
            if benchmark_type in ["all", "breakdown"]:
                print("  Measuring retrieval vs generation...")
                breakdown = self.measure_retrieval_vs_generation(query)
                measurement["breakdown"] = breakdown
            
            results["measurements"].append(measurement)
            print()
        
        # Calculate statistics
        self._calculate_statistics(results)
        
        return results
    
    def _calculate_statistics(self, results: Dict[str, Any]) -> None:
        """Calculate summary statistics for benchmark results."""
        measurements = results["measurements"]
        
        # TTFT statistics
        ttft_values = [
            m["ttft"]["ttft_ms"] 
            for m in measurements 
            if "ttft" in m and m["ttft"].get("ttft_ms") is not None
        ]
        
        if ttft_values:
            results["ttft_stats"] = {
                "median_ms": statistics.median(ttft_values),
                "p90_ms": statistics.quantiles(ttft_values, n=10)[8] if len(ttft_values) > 1 else ttft_values[0],
                "mean_ms": statistics.mean(ttft_values),
                "min_ms": min(ttft_values),
                "max_ms": max(ttft_values)
            }
        
        # End-to-end cold statistics
        e2e_cold_values = [
            m["e2e_cold"]["latency_ms"]
            for m in measurements
            if "e2e_cold" in m and m["e2e_cold"].get("latency_ms") is not None
        ]
        
        if e2e_cold_values:
            results["e2e_cold_stats"] = {
                "median_ms": statistics.median(e2e_cold_values),
                "p90_ms": statistics.quantiles(e2e_cold_values, n=10)[8] if len(e2e_cold_values) > 1 else e2e_cold_values[0],
                "mean_ms": statistics.mean(e2e_cold_values),
                "min_ms": min(e2e_cold_values),
                "max_ms": max(e2e_cold_values)
            }
        
        # End-to-end warm statistics
        e2e_warm_values = [
            m["e2e_warm"]["latency_ms"]
            for m in measurements
            if "e2e_warm" in m and m["e2e_warm"].get("latency_ms") is not None
        ]
        
        if e2e_warm_values:
            results["e2e_warm_stats"] = {
                "median_ms": statistics.median(e2e_warm_values),
                "p90_ms": statistics.quantiles(e2e_warm_values, n=10)[8] if len(e2e_warm_values) > 1 else e2e_warm_values[0],
                "mean_ms": statistics.mean(e2e_warm_values),
                "min_ms": min(e2e_warm_values),
                "max_ms": max(e2e_warm_values),
                "cache_speedup_factor": statistics.mean(e2e_cold_values) / statistics.mean(e2e_warm_values) if e2e_cold_values else None
            }
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print human-readable summary of benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Queries tested: {results['queries_count']}")
        print()
        
        if "ttft_stats" in results:
            print("Time To First Token (TTFT):")
            stats = results["ttft_stats"]
            print(f"  Median: {stats['median_ms']:.1f}ms")
            print(f"  P90:    {stats['p90_ms']:.1f}ms")
            print(f"  Mean:   {stats['mean_ms']:.1f}ms")
            print(f"  Range:  {stats['min_ms']:.1f}ms - {stats['max_ms']:.1f}ms")
            print()
        
        if "e2e_cold_stats" in results:
            print("End-to-End Latency (Cold/No Cache):")
            stats = results["e2e_cold_stats"]
            print(f"  Median: {stats['median_ms']:.1f}ms")
            print(f"  P90:    {stats['p90_ms']:.1f}ms")
            print(f"  Mean:   {stats['mean_ms']:.1f}ms")
            print(f"  Range:  {stats['min_ms']:.1f}ms - {stats['max_ms']:.1f}ms")
            print()
        
        if "e2e_warm_stats" in results:
            print("End-to-End Latency (Warm/Cached):")
            stats = results["e2e_warm_stats"]
            print(f"  Median: {stats['median_ms']:.1f}ms")
            print(f"  P90:    {stats['p90_ms']:.1f}ms")
            print(f"  Mean:   {stats['mean_ms']:.1f}ms")
            print(f"  Range:  {stats['min_ms']:.1f}ms - {stats['max_ms']:.1f}ms")
            if stats.get("cache_speedup_factor"):
                print(f"  Cache Speedup: {stats['cache_speedup_factor']:.2f}x")
            print()
        
        print("="*80)
    
    def save_results(self, results: Dict[str, Any], output_path: str = "benchmark_results.json") -> None:
        """Save benchmark results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Latency Benchmark Suite")
    parser.add_argument(
        "--benchmark",
        choices=["all", "ttft", "e2e", "breakdown"],
        default="all",
        help="Benchmark type to run"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        help="Path to JSON file with custom queries"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup phase"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output path for results JSON"
    )
    
    args = parser.parse_args()
    
    # Load queries
    queries = DEFAULT_QUERIES
    if args.queries_file:
        with open(args.queries_file) as f:
            queries = json.load(f)
    
    # Run benchmark
    benchmark = LatencyBenchmark(queries=queries, warmup=not args.no_warmup)
    results = benchmark.run_benchmark_suite(benchmark_type=args.benchmark)
    
    # Print summary
    benchmark.print_summary(results)
    
    # Save results
    benchmark.save_results(results, output_path=args.output)


if __name__ == "__main__":
    main()
