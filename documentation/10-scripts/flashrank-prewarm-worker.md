# FlashRank Prewarm Worker

Script: `scripts/flashrank_prewarm_worker.py`

## Purpose

Download and cache the FlashRank/HuggingFace model to avoid cold-start latency in production.

## Usage

```bash
python scripts/flashrank_prewarm_worker.py \
  --model ms-marco-MiniLM-L-12-v2 \
  --cache-dir data/flashrank_cache
```

## Arguments

- `--model` (string, optional): Model name (default from settings)
- `--cache-dir` (string, optional): Directory to store model cache
- `--verbose` (bool, optional): Print detailed logs

## Environment

- `.env` optional: `FLASHRANK_MODEL_NAME`, `FLASHRANK_CACHE_DIR`

## Output

- Cached model files in `data/flashrank_cache/`
- Logs: download progress, cache status

## Examples

```bash
# Prewarm default model
python scripts/flashrank_prewarm_worker.py

# Prewarm MiniLM model
python scripts/flashrank_prewarm_worker.py --model ms-marco-MiniLM-L-12-v2
```

## Deployment Notes

- Prefer running prewarm outside the web process (`FLASHRANK_PREWARM_IN_PROCESS=false`)
- Trigger during container startup or CI/CD deploy step

## Troubleshooting

- Slow download: ensure network and disk throughput
- Cache not used: verify `FLASHRANK_CACHE_DIR` is set and writable
