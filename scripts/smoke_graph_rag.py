import sys
import os
import json
import argparse

# Ensure /app is in sys.path when running inside Docker
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from rag.graph_rag import GraphRAG


def run_query(query: str, top_k: int = 5, restrict_to_context: bool = True):
    rag = GraphRAG()
    res = rag.query(query, top_k=top_k, restrict_to_context=restrict_to_context)
    out = {
        "stages": res.get("stages"),
        "retrieved": len(res.get("retrieved_chunks", [])),
        "routing": res.get("routing_result"),
        "quality_score": res.get("quality_score"),
    }
    print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Smoke test GraphRAG pipeline")
    parser.add_argument("query", nargs="?", default="How to install Carbonio?", help="Query text")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--restrict", action="store_true", default=True)
    args = parser.parse_args()
    run_query(args.query, top_k=args.top_k, restrict_to_context=args.restrict)


if __name__ == "__main__":
    main()
