#!/usr/bin/env python3
"""
Startup test suite for GraphRAG system.
Verifies all critical components are running and communicating properly.
"""

import json
import logging
import sys
import time
from typing import Dict, List, Tuple

import requests
from neo4j import GraphDatabase

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StartupTester:
    """Comprehensive startup tests for GraphRAG system."""

    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.results: List[Tuple[str, bool, str]] = []
        
    def add_result(self, test_name: str, passed: bool, message: str = ""):
        """Record test result."""
        self.results.append((test_name, passed, message))
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name} - {message}")

    def test_neo4j_connection(self) -> bool:
        """Test Neo4j database connectivity and basic stats."""
        logger.info("\n=== Testing Neo4j Connection ===")
        try:
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password)
            )
            
            with driver.session() as session:
                # Test basic query
                result = session.run("RETURN 1 as test")
                assert result.single()["test"] == 1
                
                # Get database stats
                stats = session.run("""
                    MATCH (d:Document) 
                    WITH count(d) as docs
                    MATCH (c:Chunk) 
                    WITH docs, count(c) as chunks
                    MATCH (e:Entity) 
                    WITH docs, chunks, count(e) as entities
                    MATCH ()-[r:SIMILAR_TO]->()
                    WITH docs, chunks, entities, count(r) as similarities
                    MATCH ()-[r2:RELATED_TO]->()
                    RETURN docs, chunks, entities, similarities, count(r2) as relationships
                """).single()
                
                driver.close()
                
                msg = f"Connected to Neo4j: {stats['docs']} docs, {stats['chunks']} chunks, {stats['entities']} entities"
                self.add_result("Neo4j Connection", True, msg)
                return True
                
        except Exception as e:
            self.add_result("Neo4j Connection", False, f"Error: {e}")
            return False

    def test_neo4j_indexes(self) -> bool:
        """Test that required indexes exist."""
        logger.info("\n=== Testing Neo4j Indexes ===")
        try:
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password)
            )
            
            with driver.session() as session:
                result = session.run("SHOW INDEXES")
                indexes = [record for record in result]
                
                # Check for indexes on Document(id), Chunk(id), Entity(id)
                required_labels = ["Document", "Chunk", "Entity"]
                found_labels = set()
                
                for idx in indexes:
                    # Neo4j indexes have labelsOrTypes which is a list of labels
                    labels = idx.get("labelsOrTypes") or []
                    if labels and isinstance(labels, (list, tuple)):
                        for label in labels:
                            if label in required_labels:
                                found_labels.add(label)
                
                missing = [lbl for lbl in required_labels if lbl not in found_labels]
                
                driver.close()
                
                if missing:
                    self.add_result("Neo4j Indexes", False, f"Missing indexes for: {missing}")
                    return False
                else:
                    self.add_result("Neo4j Indexes", True, f"Found indexes for {len(found_labels)} node types")
                    return True
                    
        except Exception as e:
            self.add_result("Neo4j Indexes", False, f"Error: {e}")
            return False

    def test_backend_health(self) -> bool:
        """Test backend API health endpoint."""
        logger.info("\n=== Testing Backend Health ===")
        try:
            response = requests.get(f"{self.backend_url}/api/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                msg = f"Status: {data.get('status')}, Provider: {data.get('llm_provider')}, Version: {data.get('version')}"
                self.add_result("Backend Health", True, msg)
                return True
            else:
                self.add_result("Backend Health", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.add_result("Backend Health", False, f"Error: {e}")
            return False

    def test_backend_database_stats(self) -> bool:
        """Test backend database stats endpoint."""
        logger.info("\n=== Testing Backend Database Stats ===")
        try:
            response = requests.get(f"{self.backend_url}/api/database/stats", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                msg = f"Documents: {data.get('documents')}, Chunks: {data.get('chunks')}"
                self.add_result("Backend Database Stats", True, msg)
                return True
            else:
                self.add_result("Backend Database Stats", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.add_result("Backend Database Stats", False, f"Error: {e}")
            return False

    def test_chat_tuning_config(self) -> bool:
        """Test chat tuning configuration endpoint."""
        logger.info("\n=== Testing Chat Tuning Config ===")
        try:
            response = requests.get(f"{self.backend_url}/api/chat-tuning/config", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                param_count = len(data.get('parameters', []))
                self.add_result("Chat Tuning Config", True, f"{param_count} parameters loaded")
                return True
            else:
                self.add_result("Chat Tuning Config", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.add_result("Chat Tuning Config", False, f"Error: {e}")
            return False

    def test_rag_tuning_config(self) -> bool:
        """Test RAG tuning configuration endpoint."""
        logger.info("\n=== Testing RAG Tuning Config ===")
        try:
            response = requests.get(f"{self.backend_url}/api/rag-tuning/config", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                section_count = len(data.get('sections', []))
                default_model = data.get('default_llm_model', 'unknown')
                self.add_result("RAG Tuning Config", True, f"{section_count} sections, model: {default_model}")
                return True
            else:
                self.add_result("RAG Tuning Config", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.add_result("RAG Tuning Config", False, f"Error: {e}")
            return False

    def test_chat_pipeline(self) -> bool:
        """Test end-to-end chat pipeline with SSE streaming."""
        logger.info("\n=== Testing Chat Pipeline ===")
        try:
            payload = {
                "message": "What is a graph database?",
                "session_id": "startup_test",
                "temperature": 0.7,
                "top_k": 3
            }
            
            response = requests.post(
                f"{self.backend_url}/api/chat/query",
                json=payload,
                timeout=30,
                stream=True
            )
            
            if response.status_code != 200:
                self.add_result("Chat Pipeline", False, f"HTTP {response.status_code}")
                return False
            
            # Parse SSE stream
            stages = []
            tokens = []
            quality_score = None
            sources_count = 0
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])
                        
                        if data.get('type') == 'stage':
                            stages.append(data.get('content'))
                        elif data.get('type') == 'token':
                            tokens.append(data.get('content', ''))
                        elif data.get('type') == 'quality_score':
                            quality_score = data.get('content')
                        elif data.get('type') == 'sources':
                            sources_count = len(data.get('content', []))
                            
                    except json.JSONDecodeError:
                        continue
            
            response_text = ''.join(tokens)
            
            # Validate pipeline stages
            expected_stages = ['query_analysis', 'retrieval', 'graph_reasoning', 'generation']
            missing_stages = [s for s in expected_stages if s not in stages]
            
            if missing_stages:
                self.add_result("Chat Pipeline", False, f"Missing stages: {missing_stages}")
                return False
            
            # Validate response
            if not response_text or len(response_text) < 10:
                self.add_result("Chat Pipeline", False, "Empty or too short response")
                return False
            
            msg = f"Stages: {len(stages)}, Response: {len(response_text)} chars, Sources: {sources_count}"
            if quality_score:
                msg += f", Quality: {quality_score}"
            
            self.add_result("Chat Pipeline", True, msg)
            return True
            
        except Exception as e:
            self.add_result("Chat Pipeline", False, f"Error: {e}")
            return False

    def test_frontend_availability(self) -> bool:
        """Test frontend is running and accessible."""
        logger.info("\n=== Testing Frontend Availability ===")
        try:
            response = requests.get(self.frontend_url, timeout=5)
            
            if response.status_code == 200:
                self.add_result("Frontend Availability", True, f"HTTP {response.status_code}")
                return True
            else:
                self.add_result("Frontend Availability", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.add_result("Frontend Availability", False, f"Error: {e}")
            return False

    def test_cache_system(self) -> bool:
        """Test caching system is operational."""
        logger.info("\n=== Testing Cache System ===")
        try:
            # Skip if caching disabled
            if not settings.enable_caching:
                self.add_result("Cache System", True, "Disabled in settings")
                return True
            
            # Test cache stats endpoint
            response = requests.get(f"{self.backend_url}/api/database/cache-stats", timeout=5)
            
            if response.status_code != 200:
                self.add_result("Cache System", False, f"Stats endpoint HTTP {response.status_code}")
                return False
            
            data = response.json()
            
            # Verify cache data structure
            if not isinstance(data, dict):
                self.add_result("Cache System", False, "Invalid cache stats format")
                return False
            
            # Count caches and collect stats
            cache_count = len(data)
            total_hits = sum(c.get("hits", 0) for c in data.values() if isinstance(c, dict))
            total_misses = sum(c.get("misses", 0) for c in data.values() if isinstance(c, dict))
            total_size = sum(c.get("size", 0) for c in data.values() if isinstance(c, dict))
            
            # Calculate overall hit rate
            total_requests = total_hits + total_misses
            hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
            
            msg = f"{cache_count} caches, {total_size} entries, {hit_rate:.1f}% hit rate"
            self.add_result("Cache System", True, msg)
            
            # Log individual cache stats
            for cache_name, stats in data.items():
                if isinstance(stats, dict):
                    cache_hit_rate = stats.get("hit_rate", 0.0) * 100
                    logger.info(f"  - {cache_name}: {stats.get('size', 0)} entries, "
                              f"{stats.get('hits', 0)} hits, {cache_hit_rate:.1f}% hit rate")
            
            return True
            
        except Exception as e:
            self.add_result("Cache System", False, f"Error: {e}")
            return False

    def test_settings_validation(self) -> bool:
        """Test critical settings are properly configured."""
        logger.info("\n=== Testing Settings Validation ===")
        try:
            issues = []
            
            # Check OpenAI configuration
            if settings.llm_provider == "openai":
                if not settings.openai_api_key:
                    issues.append("OpenAI API key not set")
                if not settings.openai_model:
                    issues.append("OpenAI model not set")
                else:
                    # Check for common typos
                    model = settings.openai_model.lower()
                    if "gpt-5" in model and "gpt-5" not in ["gpt-4o", "gpt-4o-mini"]:
                        issues.append(f"Suspicious model name: {settings.openai_model}")
            
            # Check Neo4j configuration
            if not settings.neo4j_uri:
                issues.append("Neo4j URI not set")
            if not settings.neo4j_username or not settings.neo4j_password:
                issues.append("Neo4j credentials not set")
            
            # Check embedding configuration
            if not settings.embedding_model:
                issues.append("Embedding model not set")
            
            if issues:
                self.add_result("Settings Validation", False, f"Issues: {', '.join(issues)}")
                return False
            else:
                msg = f"Provider: {settings.llm_provider}, Model: {settings.openai_model}, Embeddings: {settings.embedding_model}"
                self.add_result("Settings Validation", True, msg)
                return True
                
        except Exception as e:
            self.add_result("Settings Validation", False, f"Error: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all startup tests and return overall status."""
        logger.info("\n" + "="*60)
        logger.info("GRAPHRAG STARTUP TEST SUITE")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_settings_validation,
            self.test_neo4j_connection,
            self.test_neo4j_indexes,
            self.test_backend_health,
            self.test_backend_database_stats,
            self.test_cache_system,
            self.test_chat_tuning_config,
            self.test_rag_tuning_config,
            self.test_chat_pipeline,
            self.test_frontend_availability,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                self.add_result(test.__name__, False, f"Crashed: {e}")
        
        elapsed = time.time() - start_time
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for _, p, _ in self.results if p)
        failed = sum(1 for _, p, _ in self.results if not p)
        
        for test_name, passed_flag, message in self.results:
            status = "✅" if passed_flag else "❌"
            logger.info(f"{status} {test_name:<30} {message}")
        
        logger.info("="*60)
        logger.info(f"Total: {len(self.results)} tests | Passed: {passed} | Failed: {failed}")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info("="*60)
        
        return failed == 0

    def run_quick_health_check(self) -> bool:
        """Run quick health check (subset of tests for fast validation)."""
        logger.info("\n" + "="*60)
        logger.info("GRAPHRAG QUICK HEALTH CHECK")
        logger.info("="*60)
        
        tests = [
            self.test_neo4j_connection,
            self.test_backend_health,
            self.test_frontend_availability,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                self.add_result(test.__name__, False, f"Crashed: {e}")
        
        passed = sum(1 for _, p, _ in self.results if p)
        failed = sum(1 for _, p, _ in self.results if not p)
        
        logger.info("\n" + "="*60)
        logger.info(f"Quick Check: {passed}/{len(self.results)} passed")
        logger.info("="*60)
        
        return failed == 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG startup test suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick health check only (Neo4j, Backend, Frontend)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = StartupTester()
    
    if args.quick:
        success = tester.run_quick_health_check()
    else:
        success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
