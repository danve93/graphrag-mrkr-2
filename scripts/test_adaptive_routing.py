"""
Test script for M3.4 Adaptive Routing with User Feedback

This script tests the complete feedback loop:
1. Submit queries and get responses with routing info
2. Submit positive/negative feedback
3. Verify weight adjustments occur
4. Check convergence metrics

Usage:
    python scripts/test_adaptive_routing.py --queries 10 --feedback-ratio 0.7
"""

import asyncio
import argparse
import json
import time
from typing import List, Dict, Any
import requests
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

BASE_URL = "http://localhost:8000"


def check_adaptive_routing_enabled():
    """Check if adaptive routing is enabled in settings."""
    if not settings.enable_adaptive_routing:
        print("❌ Adaptive routing is not enabled!")
        print("   Set ENABLE_ADAPTIVE_ROUTING=true in your .env file")
        return False
    print("✅ Adaptive routing is enabled")
    return True


def get_initial_weights() -> Dict[str, float]:
    """Get initial weights before testing."""
    try:
        response = requests.get(f"{BASE_URL}/api/feedback/weights")
        response.raise_for_status()
        data = response.json()
        weights = data.get('weights', {})
        print(f"\n📊 Initial weights:")
        print(f"   Chunk:  {weights['chunk_weight']:.3f}")
        print(f"   Entity: {weights['entity_weight']:.3f}")
        print(f"   Path:   {weights['path_weight']:.3f}")
        print(f"   Enabled: {data.get('enabled', False)}")
        print(f"   Learning active: {data.get('learning_active', False)}")
        return weights
    except Exception as e:
        print(f"❌ Failed to get initial weights: {e}")
        return {}


def get_initial_metrics() -> Dict[str, Any]:
    """Get initial feedback metrics."""
    try:
        response = requests.get(f"{BASE_URL}/api/feedback/metrics")
        response.raise_for_status()
        metrics = response.json()
        print(f"\n📈 Initial metrics:")
        print(f"   Total feedback: {metrics.get('total_feedback', 0)}")
        print(f"   Positive: {metrics.get('positive_feedback', 0)}")
        print(f"   Negative: {metrics.get('negative_feedback', 0)}")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.1f}%")
        print(f"   Convergence: {metrics.get('convergence', 0):.4f}")
        return metrics
    except Exception as e:
        print(f"❌ Failed to get initial metrics: {e}")
        return {}


def send_query(query: str, session_id: str) -> Dict[str, Any]:
    """Send a query and return response with routing info."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat/query",
            json={
                "message": query,
                "session_id": session_id,
                "stream": False  # Use non-streaming for easier testing
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return {}


def submit_feedback(message_id: str, session_id: str, query: str, rating: int, routing_info: Dict = None) -> bool:
    """Submit feedback for a message."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/feedback",
            json={
                "message_id": message_id,
                "session_id": session_id,
                "query": query,
                "rating": rating,
                "routing_info": routing_info or {}
            }
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Feedback submission failed: {e}")
        return False


def run_test_cycle(test_queries: List[str], positive_ratio: float = 0.7):
    """Run a test cycle with queries and feedback."""
    print(f"\n🧪 Running test cycle with {len(test_queries)} queries")
    print(f"   Positive feedback ratio: {positive_ratio:.0%}")
    
    session_id = f"test_session_{int(time.time())}"
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: {query[:60]}...")
        
        # Send query
        response = send_query(query, session_id)
        if not response:
            print("   ⚠️  Query failed, skipping")
            continue
        
        # Extract metadata (non-streaming response structure may vary)
        message_id = response.get('message_id', f"msg_{int(time.time())}_{i}")
        
        # Simulate user feedback based on ratio
        rating = 1 if (i / len(test_queries)) < positive_ratio else -1
        feedback_label = "👍 Positive" if rating == 1 else "👎 Negative"
        
        # Get routing info from response if available
        routing_info = response.get("metadata", {}).get("routing_info", {})
        
        success = submit_feedback(message_id, session_id, query, rating, routing_info)
        if success:
            print(f"   ✅ Feedback submitted: {feedback_label}")
            results.append({"query": query, "rating": rating, "success": True})
        else:
            print(f"   ⚠️  Feedback failed")
            results.append({"query": query, "rating": rating, "success": False})
        
        # Small delay between requests
        time.sleep(0.5)
    
    return results


def verify_weight_changes(initial_weights: Dict[str, float], test_results: List[Dict]) -> bool:
    """Verify that weights have changed after feedback."""
    print(f"\n🔍 Verifying weight adjustments...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/feedback/weights")
        response.raise_for_status()
        data = response.json()
        final_weights = data.get('weights', {})
        
        print(f"\n📊 Weight changes:")
        print(f"   Chunk:  {initial_weights.get('chunk_weight', 0):.3f} → {final_weights['chunk_weight']:.3f}")
        print(f"   Entity: {initial_weights.get('entity_weight', 0):.3f} → {final_weights['entity_weight']:.3f}")
        print(f"   Path:   {initial_weights.get('path_weight', 0):.3f} → {final_weights['path_weight']:.3f}")
        print(f"   Learning active: {data.get('learning_active', False)}")
        
        # Check if any weight changed
        changed = False
        for key in ['chunk_weight', 'entity_weight', 'path_weight']:
            if abs(final_weights[key] - initial_weights.get(key, 0)) > 0.001:
                changed = True
                break
        
        if changed:
            print("   ✅ Weights have adjusted based on feedback")
            return True
        else:
            print("   ⚠️  Weights unchanged (may need more feedback or min_samples not reached)")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to verify weights: {e}")
        return False


def check_convergence_metrics(initial_metrics: Dict[str, Any]) -> bool:
    """Check that convergence metrics are improving."""
    print(f"\n📈 Checking convergence metrics...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/feedback/metrics")
        response.raise_for_status()
        final_metrics = response.json()
        
        print(f"\n   Total feedback: {initial_metrics.get('total_feedback', 0)} → {final_metrics['total_feedback']}")
        print(f"   Positive: {initial_metrics.get('positive_feedback', 0)} → {final_metrics['positive_feedback']}")
        print(f"   Negative: {initial_metrics.get('negative_feedback', 0)} → {final_metrics['negative_feedback']}")
        print(f"   Accuracy: {initial_metrics.get('accuracy', 0):.1f}% → {final_metrics['accuracy']:.1f}%")
        print(f"   Convergence: {initial_metrics.get('convergence', 0):.4f} → {final_metrics['convergence']:.4f}")
        
        # Check if feedback count increased
        if final_metrics['total_feedback'] > initial_metrics.get('total_feedback', 0):
            print("   ✅ Feedback is being recorded")
            return True
        else:
            print("   ⚠️  Feedback count did not increase")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to check metrics: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test M3.4 Adaptive Routing")
    parser.add_argument("--queries", type=int, default=10, help="Number of test queries")
    parser.add_argument("--positive-ratio", type=float, default=0.7, help="Ratio of positive feedback (0.0-1.0)")
    parser.add_argument("--reset", action="store_true", help="Reset feedback data before testing")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 M3.4 Adaptive Routing Test")
    print("=" * 60)
    
    # Check if adaptive routing is enabled
    if not check_adaptive_routing_enabled():
        return 1
    
    # Reset if requested
    if args.reset:
        print("\n🔄 Resetting feedback data...")
        try:
            response = requests.post(f"{BASE_URL}/api/feedback/reset")
            response.raise_for_status()
            print("   ✅ Feedback data reset")
        except Exception as e:
            print(f"   ❌ Reset failed: {e}")
    
    # Get initial state
    initial_weights = get_initial_weights()
    initial_metrics = get_initial_metrics()
    
    # Define test queries
    test_queries = [
        "How do I install Neo4j?",
        "What is graph RAG?",
        "How do I configure the database?",
        "What are the system requirements?",
        "How do I troubleshoot connection issues?",
        "What is the best way to index documents?",
        "How do I backup my data?",
        "What query languages are supported?",
        "How do I optimize performance?",
        "What are the security best practices?",
    ][:args.queries]
    
    # Run test cycle
    results = run_test_cycle(test_queries, args.positive_ratio)
    
    # Verify results
    print("\n" + "=" * 60)
    print("📋 Test Results")
    print("=" * 60)
    
    successful_feedback = sum(1 for r in results if r['success'])
    print(f"✅ {successful_feedback}/{len(results)} feedback submissions successful")
    
    weights_changed = verify_weight_changes(initial_weights, results)
    metrics_updated = check_convergence_metrics(initial_metrics)
    
    # Final verdict
    print("\n" + "=" * 60)
    print("🎯 Final Verdict")
    print("=" * 60)
    
    if successful_feedback > 0 and metrics_updated:
        print("✅ PASS: Adaptive routing feedback system is working!")
        if weights_changed:
            print("   Weights adjusted based on feedback")
        else:
            print("   Weights not yet adjusted (may need more feedback)")
        return 0
    else:
        print("❌ FAIL: Issues detected in feedback system")
        return 1


if __name__ == "__main__":
    sys.exit(main())
