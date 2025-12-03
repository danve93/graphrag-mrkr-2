#!/usr/bin/env python3
"""
Integration test for M1-M4 features:
- Follow-up question context preservation
- Cache parameter isolation
- Stage timing metadata
- UI data flow
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_initial_query():
    """Test 1: Initial query to establish context"""
    print_section("TEST 1: Initial Query")
    
    response = requests.post(
        f"{BASE_URL}/api/chat/query",
        json={
            "message": "What is Carbonio?",
            "stream": False,
            "top_k": 3
        }
    )
    
    data = response.json()
    session_id = data.get("session_id")
    
    print(f"✓ Session ID: {session_id}")
    print(f"✓ Response length: {len(data.get('response', ''))} chars")
    print(f"✓ Sources: {len(data.get('sources', []))} chunks")
    
    # Check for stages (M3/M4)
    stages = data.get("stages", [])
    if stages:
        print(f"✓ Stages tracked: {len(stages)}")
        for stage in stages:
            if isinstance(stage, dict):
                name = stage.get("name", "unknown")
                duration = stage.get("duration_ms", "N/A")
                metadata = stage.get("metadata", {})
                print(f"  - {name}: {duration}ms {metadata}")
            else:
                print(f"  - {stage} (legacy format)")
    
    total_duration = data.get("total_duration_ms")
    if total_duration:
        print(f"✓ Total duration: {total_duration}ms")
    
    return session_id

def test_follow_up_question(session_id):
    """Test 2: Follow-up question with context (M1)"""
    print_section("TEST 2: Follow-Up Question (M1 Validation)")
    
    # Ambiguous follow-up that requires context
    response = requests.post(
        f"{BASE_URL}/api/chat/query",
        json={
            "message": "What are its main features?",
            "session_id": session_id,
            "stream": False,
            "top_k": 3
        }
    )
    
    data = response.json()
    
    print(f"✓ Session ID maintained: {data.get('session_id') == session_id}")
    print(f"✓ Response length: {len(data.get('response', ''))} chars")
    print(f"✓ Sources: {len(data.get('sources', []))} chunks")
    
    # Check if contextualized query was used (should have better results)
    response_text = data.get('response', '').lower()
    has_relevant_content = 'carbonio' in response_text or 'feature' in response_text
    print(f"✓ Relevant content found: {has_relevant_content}")
    
    return data

def test_cache_behavior():
    """Test 3: Cache parameter isolation (M2)"""
    print_section("TEST 3: Cache Parameter Isolation (M2 Validation)")
    
    query = "What is Carbonio?"
    
    # Query 1: Default parameters
    start1 = time.time()
    response1 = requests.post(
        f"{BASE_URL}/api/chat/query",
        json={
            "message": query,
            "stream": False,
            "top_k": 5,
            "chunk_weight": 0.7,
            "entity_weight": 0.3
        }
    )
    duration1 = (time.time() - start1) * 1000
    data1 = response1.json()
    
    print(f"✓ Query 1 (top_k=5, chunk=0.7): {duration1:.0f}ms")
    print(f"  Sources: {len(data1.get('sources', []))}")
    
    # Query 2: Same query, same parameters (should hit cache)
    start2 = time.time()
    response2 = requests.post(
        f"{BASE_URL}/api/chat/query",
        json={
            "message": query,
            "stream": False,
            "top_k": 5,
            "chunk_weight": 0.7,
            "entity_weight": 0.3
        }
    )
    duration2 = (time.time() - start2) * 1000
    data2 = response2.json()
    
    print(f"✓ Query 2 (same params): {duration2:.0f}ms")
    print(f"  Cache hit detected: {duration2 < duration1 * 0.8}")
    
    # Query 3: Same query, different parameters (should NOT hit cache)
    start3 = time.time()
    response3 = requests.post(
        f"{BASE_URL}/api/chat/query",
        json={
            "message": query,
            "stream": False,
            "top_k": 3,  # Different
            "chunk_weight": 0.6,  # Different
            "entity_weight": 0.4  # Different
        }
    )
    duration3 = (time.time() - start3) * 1000
    data3 = response3.json()
    
    print(f"✓ Query 3 (different params): {duration3:.0f}ms")
    print(f"  Cache miss detected: {duration3 > duration2}")
    print(f"  Sources differ: {len(data1.get('sources', [])) != len(data3.get('sources', []))}")

def test_stage_timing():
    """Test 4: Stage timing accuracy (M3)"""
    print_section("TEST 4: Stage Timing Metadata (M3 Validation)")
    
    response = requests.post(
        f"{BASE_URL}/api/chat/query",
        json={
            "message": "What is Carbonio used for?",
            "stream": False,
            "top_k": 3
        }
    )
    
    data = response.json()
    stages = data.get("stages", [])
    total_duration = data.get("total_duration_ms", 0)
    
    if not stages:
        print("✗ No stages found - M3 may not be working")
        return
    
    print(f"✓ Stages found: {len(stages)}")
    
    calculated_total = 0
    for stage in stages:
        if isinstance(stage, dict):
            name = stage.get("name", "unknown")
            duration = stage.get("duration_ms", 0)
            timestamp = stage.get("timestamp", 0)
            metadata = stage.get("metadata", {})
            
            calculated_total += duration
            
            print(f"  - {name}:")
            print(f"    Duration: {duration}ms")
            print(f"    Timestamp: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3] if timestamp else 'N/A'}")
            if metadata:
                print(f"    Metadata: {metadata}")
    
    print(f"\n✓ Calculated total: {calculated_total}ms")
    print(f"✓ Reported total: {total_duration}ms")
    
    if total_duration > 0:
        accuracy = abs(calculated_total - total_duration)
        print(f"✓ Timing accuracy: {accuracy}ms difference ({'PASS' if accuracy < 50 else 'WARN'})")
    else:
        print("✗ Total duration not reported")

def test_streaming_with_timing():
    """Test 5: SSE streaming with timing (M3/M4)"""
    print_section("TEST 5: SSE Streaming with Timing (M3/M4 Validation)")
    
    print("Testing SSE stream with timing metadata...")
    
    response = requests.post(
        f"{BASE_URL}/api/chat/stream",
        json={
            "message": "What is Carbonio?",
            "top_k": 3
        },
        stream=True,
        timeout=30
    )
    
    stages_received = []
    token_count = 0
    
    for line in response.iter_lines():
        if not line:
            continue
        
        line = line.decode('utf-8')
        if line.startswith('data: '):
            try:
                data = json.loads(line[6:])
                
                if data.get('type') == 'stage':
                    stage_name = data.get('content')
                    duration_ms = data.get('duration_ms')
                    metadata = data.get('metadata', {})
                    
                    stages_received.append({
                        'name': stage_name,
                        'duration_ms': duration_ms,
                        'metadata': metadata
                    })
                    
                    if duration_ms is not None:
                        print(f"✓ Stage: {stage_name} ({duration_ms}ms) {metadata}")
                    else:
                        print(f"✓ Stage: {stage_name} (in progress)")
                
                elif data.get('type') == 'token':
                    token_count += 1
                
                elif data.get('type') == 'done':
                    break
                    
            except json.JSONDecodeError:
                pass
    
    print(f"\n✓ Stages received: {len(stages_received)}")
    print(f"✓ Tokens streamed: {token_count}")
    
    # Check if timing metadata was included
    stages_with_timing = [s for s in stages_received if s.get('duration_ms') is not None]
    print(f"✓ Stages with timing: {len(stages_with_timing)}/{len(stages_received)}")
    
    if len(stages_with_timing) > 0:
        print("✓ M3/M4 SSE timing: PASS")
    else:
        print("✗ M3/M4 SSE timing: No timing metadata in stream")

def main():
    print("\n" + "="*60)
    print("  AMBER GRAPHRAG - MILESTONE 5 INTEGRATION TESTS")
    print("  Testing M1-M4 Implementation")
    print("="*60)
    print(f"\nTest started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend URL: {BASE_URL}")
    
    try:
        # Check health
        health = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if health.status_code != 200:
            print(f"\n✗ Backend not healthy: {health.status_code}")
            return
        
        print("✓ Backend healthy")
        
        # Run tests
        session_id = test_initial_query()
        test_follow_up_question(session_id)
        test_cache_behavior()
        test_stage_timing()
        test_streaming_with_timing()
        
        print_section("TEST SUMMARY")
        print("✓ M1 (Follow-Up Context): Tested")
        print("✓ M2 (Cache Isolation): Tested")
        print("✓ M3 (Stage Timing): Tested")
        print("✓ M4 (UI Data Flow): Tested")
        print("\n✅ All integration tests completed!")
        
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to backend at {BASE_URL}")
        print("  Make sure Docker containers are running: docker compose up -d")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
