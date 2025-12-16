from rag.nodes.adaptive_router import FeedbackLearner


def test_feedback_updates_weights_with_min_samples():
    learner = FeedbackLearner(learning_rate=0.2, min_samples=1, weight_min=0.1, weight_max=0.9)

    routing_info = {"query_type": "entity_focused", "retrieval_mode": "hybrid", "routed_to": "router"}
    learner.record_feedback("What is Neo4j?", "s1", "m1", rating=1, routing_info=routing_info)

    # Entity-focused feedback should boost entity weight more than chunk
    assert learner.weights.entity_weight > learner.weights.chunk_weight
    assert learner.weights.total_feedback == 1
    assert learner.query_type_performance["entity_focused"]["positive"] == 1


def test_feedback_clamps_weights_and_tracks_negatives():
    learner = FeedbackLearner(learning_rate=0.5, min_samples=1, weight_min=0.2, weight_max=0.6)

    routing_info = {"query_type": "keyword_focused", "retrieval_mode": "hybrid", "routed_to": "router"}
    learner.record_feedback("How to configure SSL?", "s2", "m2", rating=-1, routing_info=routing_info)

    # Negative feedback should decrease chunk weight but not below weight_min
    assert learner.weights.chunk_weight >= 0.2
    assert learner.weights.negative_feedback == 1
    assert learner.routing_decisions["router"] == 1
