"""
Prometheus metric definitions for TruLens.

Defines Prometheus metrics using prometheus-client library.
These metrics are updated as TruLens records are created.
"""

from prometheus_client import Gauge, Histogram, Counter

# Answer quality metrics
answer_relevance = Gauge(
    'trulens_answer_relevance_score',
    'Answer relevance score (0-1)',
    ['retrieval_mode']
)

groundedness = Gauge(
    'trulens_groundedness_score',
    'Groundedness score (0-1)',
    ['retrieval_mode']
)

context_relevance = Gauge(
    'trulens_context_relevance_score',
    'Context relevance score (0-1)',
    ['retrieval_mode']
)

# Latency metrics
query_latency = Histogram(
    'trulens_query_latency_seconds',
    'Query latency in seconds',
    ['retrieval_mode'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Volume metrics
queries_total = Counter(
    'trulens_queries_total',
    'Total queries monitored',
    ['retrieval_mode']
)

errors_total = Counter(
    'trulens_errors_total',
    'Total errors during monitoring',
    ['error_type']
)

# Feedback evaluation duration
feedback_duration = Histogram(
    'trulens_feedback_evaluation_duration_seconds',
    'Feedback function evaluation duration',
    ['feedback_function'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
