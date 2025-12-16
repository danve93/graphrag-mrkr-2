# Evaluation System (Ragas)

## Overview

The Amber evaluation system uses [Ragas](https://github.com/explodinggradients/ragas) to measure the quality of the GraphRAG pipeline. It enables A/B testing of different configurations (vector-only vs graph-enhanced, routing on/off, etc.) and tracks key quality metrics over time.

**Key Features:**
- 🎯 **A/B Variant Testing**: Compare vector-only vs graph-enhanced retrieval
- 📊 **Comprehensive Metrics**: Context precision, recall, faithfulness, answer relevancy
- 🔄 **Per-Request Feature Flags**: Override settings without restarting the backend
- 📝 **Human-Readable Output**: Console summaries + markdown reports + JSON data
- 🛡️ **Robust Execution**: Retry logic, timeout handling, progress tracking

---

## Quick Start

### 1. Install Dependencies

```bash
uv pip install -r evals/ragas/requirements-ragas.txt
```

### 2. Prepare Dataset

Place your evaluation dataset in `evals/ragas/testsets/`. Format (CSV or JSONL):

```csv
user_input,reference,retrieved_contexts,response,metadata
"What is Carbonio?","Carbonio is an on-premises digital workplace platform...",[],,"{""intent"": ""admin"", ""source_doc"": ""admin_guide"", ""qa_type"": ""single-hop""}"
```

**Required fields:**
- `user_input`: The question to evaluate
- `reference`: Ground truth answer
- `metadata`: JSON with `intent`, `source_doc`, `qa_type`

**Optional fields:**
- `retrieved_contexts`: Pre-filled contexts (empty = runner will fetch from API)
- `response`: Pre-filled answer (empty = runner will fetch from API)

### 3. Run Evaluation

```bash
# Test vector-only baseline
uv run python evals/ragas/ragas_runner.py \
  --config evals/ragas/config.example.yaml \
  --dataset evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv \
  --variant vector_only \
  --out reports/ragas/vector_only.json \
  --progress-every 5

# Test graph-enhanced variant
uv run python evals/ragas/ragas_runner.py \
  --config evals/ragas/config.example.yaml \
  --dataset evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv \
  --variant graph_hybrid \
  --out reports/ragas/graph_hybrid.json \
  --progress-every 5
```

### 4. Compare Results

```bash
uv run python evals/ragas/ragas_report.py \
  --inputs reports/ragas/vector_only.json reports/ragas/graph_hybrid.json \
  --baseline reports/ragas/baseline.json \
  --out reports/ragas/comparison.md
```

---

## How It Works

### Architecture

The evaluation system uses **per-request feature flag overrides** to test different configurations without modifying global settings:

```
┌─────────────────┐
│  Ragas Runner   │
│                 │
│  1. Load config │
│  2. Build       │
│     payload     │
│     with eval_* │
│     flags       │
└────────┬────────┘
         │
         │ HTTP POST /api/chat/query
         │ {
         │   "message": "...",
         │   "eval_enable_query_routing": true,
         │   "eval_enable_rrf": true,
         │   ...
         │ }
         ▼
┌─────────────────┐
│  Chat Router    │
│                 │
│  1. Apply       │
│     overrides   │
│     (try block) │
│  2. Run query   │
│  3. Restore     │
│     (finally)   │
└────────┬────────┘
         │
         │ Calls graph_rag.query()
         │ with overridden settings
         ▼
┌─────────────────┐
│  RAG Pipeline   │
│                 │
│  Reads settings │
│  (temporarily   │
│   overridden)   │
└─────────────────┘
```

### Feature Flag Override Mechanism

The system adds **6 evaluation-specific fields** to the ChatRequest model:

- `eval_enable_query_routing`: Toggle query routing
- `eval_enable_structured_kg`: Toggle structured KG queries
- `eval_enable_rrf`: Toggle Reciprocal Rank Fusion
- `eval_enable_routing_cache`: Toggle routing cache
- `eval_flashrank_enabled`: Toggle FlashRank reranking
- `eval_enable_graph_clustering`: Toggle graph clustering

**How overrides work:**

1. Request arrives with `eval_*` fields set (e.g., `eval_enable_rrf: False`)
2. Chat router saves original settings: `original_rrf = settings.enable_rrf`
3. Applies override: `settings.enable_rrf = False`
4. Executes query with overridden settings
5. **Always** restores original in `finally` block: `settings.enable_rrf = original_rrf`

**Thread safety:** Settings are global, so brief overlap is possible. This is acceptable for controlled evaluation scenarios where requests are processed serially.

---

## Configuration Guide

### Config File Structure (`config.example.yaml`)

```yaml
backend:
  base_url: "http://localhost:8000"
  chat_endpoint: "/api/chat"
  timeout_seconds: 120  # Per-request timeout (default: 120s)

defaults:
  llm_model: "gpt-4o-mini"
  max_concurrency: 2  # Parallel requests (keep low to avoid timeouts)
  retry_attempts: 1   # Retry failed requests (with exponential backoff)

variants:
  vector_only:
    description: "Baseline vector retrieval"
    payload_flags:
      enable_query_routing: false
      enable_structured_kg: false
      enable_rrf: false
      flashrank_enabled: false

  graph_hybrid:
    description: "Graph-enhanced hybrid retrieval"
    payload_flags:
      enable_query_routing: true
      enable_structured_kg: true
      enable_rrf: true
      enable_routing_cache: true
      flashrank_enabled: true

paths:
  dataset: "evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv"
  output_dir: "reports/ragas"
  baseline: "reports/ragas/baseline.json"  # For regression checks

reporting:
  metrics:
    - context_precision   # Relevance of retrieved contexts
    - context_recall      # Coverage of retrieved contexts
    - faithfulness        # Answer grounded in context (critical)
    - answer_relevancy    # Answer addresses the question
```

### Adding Custom Variants

Create new variants for specific testing scenarios:

```yaml
variants:
  # Test routing effectiveness
  routing_only:
    description: "Routing enabled, other features disabled"
    payload_flags:
      enable_query_routing: true
      enable_structured_kg: false
      enable_rrf: false
      flashrank_enabled: false

  # Test reranking impact
  reranking_test:
    description: "FlashRank reranking enabled"
    payload_flags:
      enable_query_routing: false
      flashrank_enabled: true
      enable_rrf: false
```

---

## Metrics Explained

### Context Precision (Target: ≥ 0.65)

**Measures:** Relevance of retrieved contexts to the question

**Interpretation:**
- ✓ **>0.65**: Good - Retrieved contexts are relevant
- ~ **0.55-0.65**: Warning - Some irrelevant contexts retrieved
- ✗ **<0.55**: Poor - Too many irrelevant contexts

**How to improve:**
- Tune retrieval similarity thresholds
- Enable query routing to filter by category
- Adjust chunk/entity/path weights

### Context Recall (Target: ≥ 0.60)

**Measures:** Coverage of ground truth information in retrieved contexts

**Interpretation:**
- ✓ **>0.60**: Good - Retrieved contexts cover key information
- ~ **0.50-0.60**: Warning - Missing some important context
- ✗ **<0.50**: Poor - Missing critical information

**How to improve:**
- Increase `top_k` retrieval count
- Enable multi-hop reasoning for complex questions
- Check if ground truth is actually in the knowledge base

### Faithfulness (Target: ≥ 0.85) 🔴 **CRITICAL**

**Measures:** Answer grounded in retrieved contexts (not hallucinated)

**Interpretation:**
- ✓ **>0.85**: Good - Answer supported by context
- ~ **0.75-0.85**: Warning - Some unsupported claims
- ✗ **<0.75**: Poor - Significant hallucination risk

**How to improve:**
- Use more restrictive LLM prompts ("answer only from context")
- Lower LLM temperature
- Enable `restrict_to_context` mode
- **This metric should never be sacrificed for others**

### Answer Relevancy (Target: ≥ 0.80)

**Measures:** Answer addresses the user's question

**Interpretation:**
- ✓ **>0.80**: Good - Answer directly addresses question
- ~ **0.70-0.80**: Warning - Answer partially off-topic
- ✗ **<0.70**: Poor - Answer doesn't address question

**How to improve:**
- Improve query understanding (enable routing)
- Better prompt engineering
- Check if question is ambiguous

---

## Output Formats

### Console Summary

After each run, you'll see:

```
================================================================================
  RAGAS EVALUATION SUMMARY
================================================================================

Run: graph_hybrid-20250113T143022Z
Variant: graph_hybrid
Status: EVALUATED

Execution:
  Total: 60 | Evaluated: 58 | Errors: 2

Metrics:
  answer_relevancy         : 0.834 ✓
  context_precision        : 0.712 ✓
  context_recall           : 0.645 ✓
  faithfulness             : 0.891 ✓

================================================================================
```

### Markdown Report (`.md` file)

Human-readable report with:
- **Configuration**: Variant flags used
- **Execution Summary**: Total samples, errors, completion rate
- **Metrics Table**: All metrics with status indicators
- **Sample Errors**: First 5 errors encountered
- **Recommendations**: Automatic suggestions based on thresholds

Example recommendation:

```markdown
## Recommendations

- ⚠️  Context recall is low - retrieval may be missing key information
- ✅ Faithfulness meets target threshold
```

### JSON Report (`.json` file)

Programmatic access with:
- Full run metadata (git commit, dataset hash, timestamp)
- Per-sample results with contexts and answers
- Aggregate metrics
- Error details

---

## Best Practices

### Dataset Design

**1. Balanced Coverage**
- Include both simple and complex questions
- Mix single-hop and multi-hop queries
- Cover all document types (admin, user)

**2. Quality Ground Truth**
- Reference answers should be concise but complete
- Mark expected source documents in metadata
- Use diverse question types (factual, procedural, conceptual)

**3. Metadata Fields**
```json
{
  "intent": "admin",           // admin | user
  "source_doc": "admin_guide", // Expected source document
  "source_pages": "[5,6]",     // Expected pages
  "qa_type": "multi-hop"       // single-hop | multi-hop
}
```

### Running Evaluations

**1. Start Small**
```yaml
defaults:
  max_concurrency: 2  # Low concurrency prevents timeouts
  max_examples: 10    # Test with subset first
```

**2. Monitor Progress**
```bash
--progress-every 5        # Print after every 5 samples
--progress-interval 10    # Print heartbeat every 10 seconds
```

**3. Handle Errors**
- Check `sample_errors` in output for common failures
- Increase `timeout_seconds` if seeing timeouts
- Increase `retry_attempts` for flaky connections

### Establishing Baselines

**1. Create Baseline**

Run your best configuration and save as baseline:

```bash
cp reports/ragas/graph_hybrid.json reports/ragas/baseline.json
```

**2. Regression Testing**

Compare new runs against baseline:

```bash
uv run python evals/ragas/ragas_report.py \
  --inputs reports/ragas/new_config.json \
  --baseline reports/ragas/baseline.json \
  --out reports/ragas/regression_check.md
```

**3. Regression Threshold**

The reporter flags regressions **>5% drop** from baseline:

```markdown
## Regression Alerts (>-5% drop)
- faithfulness: -6.2% vs baseline  🔴 CRITICAL
```

---

## Troubleshooting

### "Variant flags not taking effect"

**Symptom:** Both vector_only and graph_hybrid produce same results

**Cause:** Ensure backend is using the updated code with eval_* fields

**Solution:**
1. Restart backend: `docker-compose restart backend`
2. Check logs for `[Eval Override]` messages
3. Verify API accepts eval_* fields: check API schema

### "Timeout errors"

**Symptom:** Many requests fail with timeout

**Solutions:**
```yaml
backend:
  timeout_seconds: 180  # Increase from 120

defaults:
  max_concurrency: 1    # Decrease from 2
```

### "Missing contexts"

**Symptom:** `missing_contexts: 60` in output

**Cause:** Backend not returning `sources` field or sources empty

**Solutions:**
1. Check backend logs for retrieval failures
2. Verify knowledge base has relevant documents
3. Lower `min_retrieval_similarity` threshold

### "Ragas metrics failed"

**Symptom:** `status: metrics_failed`

**Cause:** Missing dependencies or API quota exceeded

**Solutions:**
```bash
# Reinstall dependencies
uv pip install -r evals/ragas/requirements-ragas.txt

# Check for rate limit errors in sample_errors
```

---

## Finding the Sweet Spot: Configuration Optimization

### Overview

Use Ragas to systematically find optimal RAG settings through A/B testing and parameter sweeps. This section shows how to use the evaluation system for data-driven configuration tuning.

### Strategy: Systematic Testing

**Principle:** Change one thing at a time, measure impact, iterate.

**Priority order:**
1. **Faithfulness first** (>= 0.85) - Never sacrifice this for other metrics
2. **Answer relevancy** (>= 0.80) - Ensure users get relevant answers
3. **Balance precision/recall** - Both above thresholds (0.65/0.60)

### Step 1: Test Feature Impact

Create variants to test each feature in isolation:

```yaml
# evals/ragas/config.example.yaml

variants:
  baseline:
    description: "Current production settings"
    payload_flags:
      enable_query_routing: true
      enable_structured_kg: true
      enable_rrf: true
      flashrank_enabled: true

  no_routing:
    description: "Measure routing impact"
    payload_flags:
      enable_query_routing: false  # Only change this
      enable_structured_kg: true
      enable_rrf: true
      flashrank_enabled: true

  no_reranking:
    description: "Measure FlashRank impact"
    payload_flags:
      enable_query_routing: true
      enable_structured_kg: true
      enable_rrf: true
      flashrank_enabled: false  # Only change this

  no_rrf:
    description: "Measure RRF impact"
    payload_flags:
      enable_query_routing: true
      enable_structured_kg: true
      enable_rrf: false  # Only change this
      flashrank_enabled: true
```

**Run all variants:**

```bash
#!/bin/bash
# test_feature_impact.sh

VARIANTS=("baseline" "no_routing" "no_reranking" "no_rrf")

for variant in "${VARIANTS[@]}"; do
  echo "Testing: $variant"
  uv run python evals/ragas/ragas_runner.py \
    --config evals/ragas/config.example.yaml \
    --dataset evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv \
    --variant $variant \
    --out "reports/ragas/${variant}.json"
done

# Compare all results
uv run python evals/ragas/ragas_report.py \
  --inputs reports/ragas/baseline.json \
           reports/ragas/no_routing.json \
           reports/ragas/no_reranking.json \
           reports/ragas/no_rrf.json \
  --baseline reports/ragas/baseline.json \
  --out reports/ragas/feature_impact.md
```

**Analyze results:**

```markdown
## Metrics Comparison

| Metric            | baseline | no_routing | no_reranking | no_rrf |
|-------------------|----------|------------|--------------|--------|
| faithfulness      | 0.891    | 0.883 (-1%)| 0.875 (-2%) | 0.888  |
| context_precision | 0.712    | 0.651 (-9%)| 0.698 (-2%) | 0.705  |
| context_recall    | 0.645    | 0.589 (-9%)| 0.632 (-2%) | 0.641  |
| answer_relevancy  | 0.834    | 0.798 (-4%)| 0.827 (-1%) | 0.831  |

## Conclusions
- Routing has biggest impact on precision (-9%)
- Reranking affects faithfulness most (-2%)
- RRF provides marginal gains
- Keep: routing, reranking; Consider removing: RRF (if latency matters)
```

### Step 2: Test Retrieval Modes

Compare different retrieval strategies:

```yaml
variants:
  simple_vector:
    description: "Pure vector search"
    retrieval_mode: "simple"
    payload_flags: { ... all false ... }

  hybrid_balanced:
    description: "Hybrid retrieval"
    retrieval_mode: "hybrid"
    payload_flags: { ... }

  entity_focused:
    description: "Entity-based retrieval"
    retrieval_mode: "entity_only"
    payload_flags: { ... }
```

### Step 3: Parameter Tuning

Test different values for key parameters:

```yaml
# Variant A: More contexts
variant_topk_10:
  defaults:
    top_k: 10
    chunk_weight: 0.4
    entity_weight: 0.4
    path_weight: 0.2

# Variant B: Fewer, precise contexts
variant_topk_3:
  defaults:
    top_k: 3
    chunk_weight: 0.6
    entity_weight: 0.3
    path_weight: 0.1
```

### Step 4: Grid Search (Advanced)

For systematic parameter optimization:

```python
#!/usr/bin/env python3
# optimize_parameters.py

import subprocess
import json
from pathlib import Path
import pandas as pd

# Parameters to test
param_grid = {
    'top_k': [3, 5, 10],
    'chunk_weight': [0.3, 0.4, 0.5, 0.6],
    'flashrank_enabled': [True, False],
}

results = []

for top_k in param_grid['top_k']:
    for chunk_weight in param_grid['chunk_weight']:
        for flashrank in param_grid['flashrank_enabled']:
            config_name = f"topk{top_k}_chunk{chunk_weight}_flash{flashrank}"

            print(f"Testing: {config_name}")

            # Create variant config dynamically
            variant_config = {
                'description': config_name,
                'payload_flags': {
                    'flashrank_enabled': flashrank,
                    # ... other flags
                }
            }

            # Run evaluation
            output = f"reports/ragas/grid/{config_name}.json"
            subprocess.run([
                "uv", "run", "python", "evals/ragas/ragas_runner.py",
                "--config", "evals/ragas/config.example.yaml",
                "--dataset", "evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv",
                "--variant", "baseline",  # Use baseline with defaults
                "--out", output
            ])

            # Load results
            with open(output) as f:
                data = json.load(f)
                metrics = data.get("metrics", {}).get("aggregate", {})

                results.append({
                    'config': config_name,
                    'top_k': top_k,
                    'chunk_weight': chunk_weight,
                    'flashrank': flashrank,
                    'faithfulness': metrics.get('faithfulness', 0),
                    'precision': metrics.get('context_precision', 0),
                    'recall': metrics.get('context_recall', 0),
                    'relevancy': metrics.get('answer_relevancy', 0),
                })

# Analyze results
df = pd.DataFrame(results)

# Find best configuration
# Weighted score: prioritize faithfulness
df['score'] = (
    df['faithfulness'] * 0.4 +
    df['relevancy'] * 0.3 +
    df['precision'] * 0.2 +
    df['recall'] * 0.1
)

best = df.loc[df['score'].idxmax()]
print("\n🎯 Best Configuration:")
print(f"  top_k: {best['top_k']}")
print(f"  chunk_weight: {best['chunk_weight']}")
print(f"  flashrank: {best['flashrank']}")
print(f"  Score: {best['score']:.3f}")
print(f"\n  Metrics:")
print(f"    Faithfulness: {best['faithfulness']:.3f}")
print(f"    Precision: {best['precision']:.3f}")
print(f"    Recall: {best['recall']:.3f}")
print(f"    Relevancy: {best['relevancy']:.3f}")

# Save full results
df.to_csv('reports/ragas/grid_search_results.csv', index=False)
print(f"\n✅ Full results saved to: reports/ragas/grid_search_results.csv")
```

### Optimization Workflow (4-Week Plan)

**Week 1: Feature Impact**
```bash
# Test each feature in isolation
./test_feature_impact.sh
# Decision: Keep high-impact features, remove low-impact ones
```

**Week 2: Retrieval Mode**
```bash
# Compare retrieval strategies
./test_retrieval_modes.sh
# Decision: Choose best mode for your use case
```

**Week 3: Parameter Tuning**
```bash
# Fine-tune top_k, weights, thresholds
python optimize_parameters.py
# Decision: Set optimal parameter values
```

**Week 4: Validation**
```bash
# Run winning config on full dataset
uv run python evals/ragas/ragas_runner.py \
  --config evals/ragas/config_optimized.yaml \
  --dataset evals/ragas/testsets/full_dataset.csv \
  --variant optimized \
  --out reports/ragas/optimized.json

# Establish as new baseline
cp reports/ragas/optimized.json reports/ragas/baseline.json
```

### Quick Wins: Common Optimizations

**1. Increase Faithfulness (Reduce Hallucinations)**
```yaml
# More restrictive generation
defaults:
  temperature: 0.3  # Lower from 0.7
  restrict_to_context: true

payload_flags:
  flashrank_enabled: true  # Better context ranking
```

**2. Improve Precision (Reduce Noise)**
```yaml
defaults:
  top_k: 3  # Fewer, better contexts

payload_flags:
  enable_query_routing: true  # Filter by category
  flashrank_enabled: true     # Rerank for relevance
```

**3. Improve Recall (Reduce Missed Information)**
```yaml
defaults:
  top_k: 10  # More contexts
  use_multi_hop: true  # Follow entity relationships

payload_flags:
  enable_rrf: true  # Combine multiple retrieval methods
```

### Tracking Progress

**1. Version Control Your Configs**
```bash
git add evals/ragas/config.example.yaml
git commit -m "eval: top_k=5 improves recall by 12%"
```

**2. Maintain Optimization Log**
```markdown
# OPTIMIZATION_LOG.md

## 2025-01-13: Initial Baseline
- Config: default settings
- Faithfulness: 0.821
- Precision: 0.623
- Recall: 0.512

## 2025-01-14: Enable Routing
- Change: enable_query_routing: true
- Result: +8.6% precision, +7.3% recall
- Decision: Keep enabled

## 2025-01-15: Tune top_k
- Tested: 3, 5, 10, 15
- Winner: top_k=5
- Result: +5.2% faithfulness, balanced precision/recall
- Decision: Use top_k=5
```

**3. Store All Results**
```bash
mkdir -p reports/ragas/history/$(date +%Y-%m-%d)/
cp reports/ragas/*.json reports/ragas/history/$(date +%Y-%m-%d)/
```

### Continuous Optimization

**Monthly Review**
```bash
# Compare current vs baseline
uv run python evals/ragas/ragas_runner.py \
  --variant current_prod \
  --out reports/ragas/monthly_check.json

uv run python evals/ragas/ragas_report.py \
  --inputs reports/ragas/monthly_check.json \
  --baseline reports/ragas/baseline.json \
  --out reports/ragas/monthly_regression_check.md
```

**Regression Alerts**
Set up alerts if any metric drops >5%:
```python
# check_regression.py
import json

with open('reports/ragas/monthly_check.json') as f:
    current = json.load(f)
with open('reports/ragas/baseline.json') as f:
    baseline = json.load(f)

current_metrics = current['metrics']['aggregate']
baseline_metrics = baseline['metrics']['aggregate']

for metric in ['faithfulness', 'context_precision', 'context_recall', 'answer_relevancy']:
    curr = current_metrics.get(metric, 0)
    base = baseline_metrics.get(metric, 0)
    delta = (curr - base) / base if base else 0

    if delta < -0.05:  # >5% drop
        print(f"⚠️  REGRESSION: {metric} dropped {delta*100:.1f}%")
        # Send alert (email, Slack, etc.)
```

---

## Advanced Usage

### Custom Metrics

Add optional metrics in config:

```yaml
reporting:
  metrics:
    - context_precision
    - context_recall
    - faithfulness
    - answer_relevancy
    - context_entities_recall   # Entity coverage
    - factual_correctness       # Strict fact checking
```

**Note:** `factual_correctness` requires additional LLM calls and is more expensive.

### Environment Variables

Set API key for backend auth:

```yaml
backend:
  api_key_env: "AMBER_API_KEY"  # Reads from environment
```

Then: `export AMBER_API_KEY="your-key-here"`

### Parallel Variant Testing

Run multiple variants simultaneously:

```bash
# Terminal 1
uv run python evals/ragas/ragas_runner.py \
  --variant vector_only --out reports/ragas/v1.json &

# Terminal 2
uv run python evals/ragas/ragas_runner.py \
  --variant graph_hybrid --out reports/ragas/v2.json &

# Wait for both
wait
```

**Warning:** Concurrent requests may see brief settings overlap. Use serial execution for critical evaluations.

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Ragas Evaluation

on:
  pull_request:
    paths:
      - 'rag/**'
      - 'api/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Start services
        run: docker-compose up -d

      - name: Wait for backend
        run: ./scripts/wait-for-backend.sh

      - name: Run evaluation
        run: |
          uv pip install -r evals/ragas/requirements-ragas.txt
          uv run python evals/ragas/ragas_runner.py \
            --config evals/ragas/config.example.yaml \
            --dataset evals/ragas/testsets/test_mini.jsonl \
            --variant graph_hybrid \
            --out reports/ragas/pr_${GITHUB_PR_NUMBER}.json

      - name: Check regression
        run: |
          uv run python evals/ragas/ragas_report.py \
            --inputs reports/ragas/pr_${GITHUB_PR_NUMBER}.json \
            --baseline reports/ragas/baseline.json \
            --out reports/ragas/regression.md

      - name: Comment PR
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('reports/ragas/regression.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

---

## Removal

To completely remove the evaluation system:

```bash
# Delete evaluation code
rm -rf evals/ragas/

# Delete reports
rm -rf reports/ragas/

# Delete tests
rm tests/unit/test_eval_overrides.py
rm tests/integration/test_ragas_payload.py
```

**No core files are modified** - the system is fully self-contained.

---

## References

- [Ragas Documentation](https://docs.ragas.io/)
- [Ragas Metrics Guide](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
- [Configuration Examples](../../evals/ragas/config.example.yaml)
- [Original Plan](../../docs/ragas-evaluation-plan.md)
