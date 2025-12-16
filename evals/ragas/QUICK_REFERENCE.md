# Ragas Evaluation - Quick Reference

## One-Command Evaluation

```bash
# Vector baseline
uv run python evals/ragas/ragas_runner.py \
  --config evals/ragas/config.example.yaml \
  --dataset evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv \
  --variant vector_only \
  --out reports/ragas/vector_only.json

# Graph hybrid
uv run python evals/ragas/ragas_runner.py \
  --config evals/ragas/config.example.yaml \
  --dataset evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv \
  --variant graph_hybrid \
  --out reports/ragas/graph_hybrid.json

# Compare
uv run python evals/ragas/ragas_report.py \
  --inputs reports/ragas/vector_only.json reports/ragas/graph_hybrid.json \
  --out reports/ragas/comparison.md
```

## Metric Targets

| Metric | Target | Critical |
|--------|--------|----------|
| **Faithfulness** | ≥ 0.85 | 🔴 YES - Never sacrifice |
| Context Precision | ≥ 0.65 | ⚠️ Important |
| Context Recall | ≥ 0.60 | ⚠️ Important |
| Answer Relevancy | ≥ 0.80 | ⚠️ Important |

## Output Files

Each run creates:
- `{variant}.json` - Full results (programmatic)
- `{variant}.md` - Human-readable summary
- Console output - Quick overview

## Common Options

```bash
--progress-every 5         # Print every 5 samples
--progress-interval 10     # Heartbeat every 10 seconds
```

## Quick Troubleshooting

**Timeouts?**
```yaml
backend:
  timeout_seconds: 180  # Increase
defaults:
  max_concurrency: 1    # Decrease
```

**No contexts retrieved?**
- Check backend is running
- Verify knowledge base has documents
- Check `sample_errors` in JSON output

**Variants produce same results?**
- Restart backend
- Check logs for `[Eval Override]` messages
- Verify eval_* fields in ChatRequest

## Variant Configurations

**vector_only**: All features disabled (baseline)
```yaml
enable_query_routing: false
enable_structured_kg: false
enable_rrf: false
flashrank_enabled: false
```

**graph_hybrid**: All features enabled
```yaml
enable_query_routing: true
enable_structured_kg: true
enable_rrf: true
enable_routing_cache: true
flashrank_enabled: true
```

## Full Documentation

See [documentation/08-operations/evaluation-system.md](../../documentation/08-operations/evaluation-system.md)
