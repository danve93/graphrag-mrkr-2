"""
Ragas evaluation runner (scaffold).

Design goals:
- Self-contained and removable (lives in evals/ragas, no changes to core settings).
- Works with uv: `uv run python evals/ragas/ragas_runner.py ...`
- Safe defaults; fails fast when required data is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    return records


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_variant(config: Dict[str, Any], variant: str) -> Dict[str, Any]:
    variants = config.get("variants") or {}
    if variant not in variants:
        raise KeyError(f"Variant '{variant}' not found in config. Available: {list(variants.keys())}")
    return variants[variant]


def sanitize_contexts(raw_contexts: Any) -> List[str]:
    if raw_contexts is None:
        return []
    if isinstance(raw_contexts, list):
        return [str(c) for c in raw_contexts if c is not None]
    if isinstance(raw_contexts, str):
        try:
            loaded = json.loads(raw_contexts)
            if isinstance(loaded, list):
                return [str(c) for c in loaded if c is not None]
        except json.JSONDecodeError:
            return [raw_contexts]
    return []


def parse_metadata(meta: Any) -> Dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, str):
        try:
            loaded = json.loads(meta)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {"raw": meta}
    return {}


def extract_contexts_from_sources(sources: Any) -> Tuple[List[str], List[str]]:
    if not sources:
        return [], []
    contexts: List[str] = []
    doc_ids: set[str] = set()
    for src in sources:
        if not isinstance(src, dict):
            continue
        content = str(src.get("content", "")).strip()
        doc = src.get("document_name") or src.get("document_id") or src.get("doc_id")
        citation = src.get("citation") or src.get("chunk_id")
        prefix_parts = []
        if doc:
            prefix_parts.append(str(doc))
            doc_ids.add(str(doc))
        if citation:
            prefix_parts.append(str(citation))
        prefix = " | ".join(prefix_parts)
        if prefix:
            contexts.append(f"{prefix}: {content}")
        else:
            contexts.append(content)
    return contexts, sorted(doc_ids)


def try_get_git_commit() -> Optional[str]:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def compute_ragas_metrics(
    samples: List[Dict[str, Any]], metric_names: List[str]
) -> Optional[Dict[str, Any]]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.metrics.critique import factual_correctness
        from ragas.metrics.relational import context_entities_recall
    except Exception as exc:  # pragma: no cover - optional dependency guard
        print(
            f"[WARN] Ragas or datasets not installed. Install optional deps via "
            f"'uv pip install -r evals/ragas/requirements-ragas.txt'. Error: {exc}",
            file=sys.stderr,
        )
        return None

    metric_map = {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_entities_recall": context_entities_recall,
        "factual_correctness": factual_correctness,
    }

    selected_metrics = []
    for name in metric_names:
        metric = metric_map.get(name)
        if metric is None:
            print(f"[WARN] Metric '{name}' not recognized; skipping.", file=sys.stderr)
            continue
        selected_metrics.append(metric)

    if not selected_metrics:
        print("[WARN] No valid metrics selected; skipping evaluation.", file=sys.stderr)
        return None

    ds = Dataset.from_list(samples)
    result = evaluate(ds, metrics=selected_metrics)
    return {
        "aggregate": {k: float(v) for k, v in result.items() if k != "result"},
        "samples": result.get("result"),
    }


def build_chat_payload(
    record: Dict[str, Any],
    defaults: Dict[str, Any],
    variant_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    message = record.get("user_input") or record.get("question") or ""
    variant_flags = variant_cfg.get("payload_flags") or {}

    payload: Dict[str, Any] = {
        "message": message,
        "session_id": str(uuid.uuid4()),
        "stream": False,
    }

    # Optional knobs (respect ChatRequest schema)
    retrieval_mode = variant_cfg.get("retrieval_mode") or defaults.get("retrieval_mode") or "hybrid"
    if retrieval_mode:
        payload["retrieval_mode"] = retrieval_mode

    for key in [
        "top_k",
        "temperature",
        "use_multi_hop",
        "chunk_weight",
        "entity_weight",
        "path_weight",
        "max_hops",
        "beam_size",
        "graph_expansion_depth",
        "restrict_to_context",
        "llm_model",
        "embedding_model",
    ]:
        val = defaults.get(key)
        if val is not None:
            payload[key] = val

    # Allow variant overrides for recognized fields
    for key in ["retrieval_mode", "top_k", "temperature", "use_multi_hop"]:
        if key in variant_flags:
            payload[key] = variant_flags[key]

    # Map variant feature flags to eval_* API fields
    flag_mapping = {
        "enable_query_routing": "eval_enable_query_routing",
        "enable_structured_kg": "eval_enable_structured_kg",
        "enable_rrf": "eval_enable_rrf",
        "enable_routing_cache": "eval_enable_routing_cache",
        "flashrank_enabled": "eval_flashrank_enabled",
        "enable_graph_clustering": "eval_enable_graph_clustering",
    }

    for old_key, new_key in flag_mapping.items():
        if old_key in variant_flags:
            payload[new_key] = variant_flags[old_key]

    return payload


async def fetch_backend_answers(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
    variant_cfg: Dict[str, Any],
    progress_every: int = 0,
    progress_interval_seconds: int = 5,
) -> List[Dict[str, Any]]:
    try:
        import httpx
    except Exception as exc:  # pragma: no cover - optional dependency guard
        print(
            f"[WARN] httpx not installed; install eval deps via 'uv pip install -r evals/ragas/requirements-ragas.txt'. Error: {exc}",
            file=sys.stderr,
        )
        return records

    backend = config.get("backend") or {}
    defaults = config.get("defaults") or {}
    base_url = backend.get("base_url", "http://localhost:8000")
    chat_endpoint = backend.get("chat_endpoint", "/api/chat")
    timeout_seconds = backend.get("timeout_seconds", 120)
    max_concurrency = defaults.get("max_concurrency", 4) or 4

    api_key_env = backend.get("api_key_env")
    headers = {}
    if api_key_env and os.getenv(api_key_env):
        headers["Authorization"] = f"Bearer {os.getenv(api_key_env)}"

    # Inject OpenTelemetry context for distributed tracing
    try:
        from evals.ragas.otel_ragas import inject_trace_context
        inject_trace_context(headers)
    except Exception:
        pass

    url = f"{base_url.rstrip('/')}{chat_endpoint.rstrip('/')}/query"

    # Use structured timeout with separate connect/read timeouts
    timeout_config = httpx.Timeout(
        connect=10.0,
        read=timeout_seconds,
        write=10.0,
        pool=10.0
    )

    sem = asyncio.Semaphore(max_concurrency)
    updated: List[Dict[str, Any]] = []

    total = len(records)
    stats = {"completed": 0, "errors": 0, "started": 0}
    error_samples: List[str] = []
    lock = asyncio.Lock()

    async def report_progress() -> None:
        if progress_every <= 0:
            return
        async with lock:
            stats["completed"] += 1
            if stats["completed"] == total or stats["completed"] % progress_every == 0:
                extra = f"; sample error: {error_samples[0]}" if error_samples else ""
                print(
                    f"[progress] completed {stats['completed']}/{total} "
                    f"(errors: {stats['errors']}, started: {stats['started']}){extra}",
                    flush=True,
                )

    async def progress_ticker(stop_event: asyncio.Event) -> None:
        if progress_interval_seconds <= 0:
            return
        while not stop_event.is_set():
            await asyncio.sleep(progress_interval_seconds)
            async with lock:
                extra = f"; sample error: {error_samples[0]}" if error_samples else ""
                print(
                    f"[progress] started {stats['started']}/{total}, "
                    f"completed {stats['completed']}, errors {stats['errors']}{extra}",
                    flush=True,
                )
        async with lock:
            print(
                f"[progress] finished {stats['completed']}/{total} "
                f"with errors {stats['errors']}",
                flush=True,
            )

    stop_event = asyncio.Event()

    async with httpx.AsyncClient(timeout=timeout_config, headers=headers) as client:
        ticker_task = asyncio.create_task(progress_ticker(stop_event))

        async def _call_one(idx: int, rec: Dict[str, Any]) -> Dict[str, Any]:
            payload = build_chat_payload(rec, defaults, variant_cfg)
            if not payload.get("message"):
                rec["__error"] = "empty message"
                await report_progress()
                return rec

            # Get retry configuration
            retry_attempts = defaults.get("retry_attempts", 1) or 1
            last_error = None

            for attempt in range(retry_attempts):
                try:
                    async with sem:
                        async with lock:
                            stats["started"] += 1
                        resp = await client.post(url, json=payload)

                    if resp.status_code != 200:
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
                        if attempt < retry_attempts - 1:
                            # Exponential backoff: 1s, 2s, 4s...
                            await asyncio.sleep(2 ** attempt)
                            print(f"[retry] Attempt {attempt+1}/{retry_attempts} after error: {last_error}", file=sys.stderr, flush=True)
                            continue
                        # Final attempt failed
                        rec["__error"] = last_error
                        async with lock:
                            stats["errors"] += 1
                            if len(error_samples) < 3:
                                error_samples.append(last_error)
                        await report_progress()
                        return rec

                    # Success - parse response
                    data = resp.json()
                    break

                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 ** attempt)
                        print(f"[retry] Attempt {attempt+1}/{retry_attempts} after exception: {last_error}", file=sys.stderr, flush=True)
                        continue
                    # Final attempt failed
                    err_msg = f"Request failed after {retry_attempts} attempts: {last_error}"
                    rec["__error"] = err_msg
                    async with lock:
                        stats["errors"] += 1
                        if len(error_samples) < 3:
                            error_samples.append(rec["__error"])
                    await report_progress()
                    return rec

            sources = data.get("sources") or []
            contexts, doc_ids = extract_contexts_from_sources(sources)

            base_meta = parse_metadata(rec.get("metadata"))
            base_meta["retrieved_docs"] = doc_ids
            base_meta["variant_flags"] = variant_cfg.get("payload_flags") or {}
            base_meta["api_variant"] = variant_cfg.get("description") or variant_cfg.get("name")

            rec["response"] = data.get("message", "")
            rec["retrieved_contexts"] = contexts
            rec["metadata"] = base_meta
            rec.pop("__error", None)
            await report_progress()
            return rec

        tasks = [_call_one(i, dict(rec)) for i, rec in enumerate(records)]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Handle any exceptions that occurred during gather
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with exception: {result}")
                    records[i]["__error"] = f"Exception: {type(result).__name__}: {result}"
        finally:
            stop_event.set()
            await ticker_task
    updated.extend(results)
    return updated


def prepare_samples(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    missing_answer = 0
    missing_contexts = 0

    for rec in records:
        question = rec.get("user_input") or rec.get("question") or ""
        ground_truth = rec.get("reference") or rec.get("ground_truth") or ""
        answer = rec.get("response") or rec.get("answer") or ""
        contexts = sanitize_contexts(rec.get("retrieved_contexts") or rec.get("contexts"))
        metadata = parse_metadata(rec.get("metadata"))

        if not answer:
            missing_answer += 1
        if not contexts:
            missing_contexts += 1

        samples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "metadata": metadata,
            }
        )

    return {
        "samples": samples,
        "missing_answer": missing_answer,
        "missing_contexts": missing_contexts,
    }


def format_metric_value(value: Any, metric_name: str) -> str:
    """Format a metric value with appropriate color coding."""
    if value is None:
        return "n/a"

    try:
        val = float(value)
    except (ValueError, TypeError):
        return str(value)

    # Define thresholds for different metrics
    good_thresholds = {
        "context_precision": 0.65,
        "context_recall": 0.60,
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_entities_recall": 0.70,
        "factual_correctness": 0.75,
    }

    threshold = good_thresholds.get(metric_name, 0.70)
    formatted = f"{val:.3f}"

    if val >= threshold:
        return f"{formatted} ✓"  # Good
    elif val >= threshold * 0.85:
        return f"{formatted} ~"  # Warning
    else:
        return f"{formatted} ✗"  # Poor

    return formatted


def format_run_summary_markdown(run_payload: Dict[str, Any]) -> str:
    """Format run results as human-readable markdown."""
    lines: List[str] = []

    lines.append("# Ragas Evaluation Run Summary")
    lines.append("")
    lines.append(f"**Run ID**: `{run_payload.get('run_id', 'unknown')}`  ")
    lines.append(f"**Variant**: `{run_payload.get('variant', 'unknown')}`  ")
    lines.append(f"**Timestamp**: {run_payload.get('timestamp_utc', 'unknown')}  ")
    lines.append(f"**Status**: {run_payload.get('status', 'unknown')}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    variant_flags = run_payload.get("variant_flags", {})
    if variant_flags:
        lines.append("**Variant Flags**:")
        for key, val in variant_flags.items():
            lines.append(f"- `{key}`: {val}")
    else:
        lines.append("No variant flags configured")
    lines.append("")

    # Execution summary
    counts = run_payload.get("counts", {})
    lines.append("## Execution Summary")
    lines.append("")
    lines.append(f"- **Total samples**: {counts.get('total', 0)}")
    lines.append(f"- **Successfully evaluated**: {counts.get('evaluated', 0)}")
    lines.append(f"- **Missing answers**: {counts.get('missing_answer', 0)}")
    lines.append(f"- **Missing contexts**: {counts.get('missing_contexts', 0)}")
    lines.append(f"- **Errors**: {counts.get('errors', 0)}")
    lines.append("")

    # Sample errors if any
    sample_errors = run_payload.get("sample_errors", [])
    if sample_errors:
        lines.append("## Sample Errors")
        lines.append("")
        for i, err in enumerate(sample_errors[:5], 1):
            lines.append(f"{i}. `{err}`")
        lines.append("")

    # Metrics
    metrics = run_payload.get("metrics", {})
    if metrics and metrics.get("aggregate"):
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Score | Status |")
        lines.append("|--------|-------|--------|")

        aggregate = metrics.get("aggregate", {})
        for metric, value in sorted(aggregate.items()):
            formatted = format_metric_value(value, metric)
            # Extract just the number for the score column
            score_str = f"{value:.3f}" if value is not None else "n/a"
            # Get the status indicator
            if "✓" in formatted:
                status = "✓ Good"
            elif "~" in formatted:
                status = "~ Warning"
            elif "✗" in formatted:
                status = "✗ Poor"
            else:
                status = "-"
            lines.append(f"| {metric} | {score_str} | {status} |")
        lines.append("")

    # Notes
    notes = run_payload.get("notes", [])
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if run_payload.get("status") == "evaluated" and metrics and metrics.get("aggregate"):
        agg = metrics.get("aggregate", {})
        recommendations = []

        if agg.get("context_precision", 0) < 0.65:
            recommendations.append("⚠️  Context precision is low - consider improving retrieval relevance")
        if agg.get("context_recall", 0) < 0.60:
            recommendations.append("⚠️  Context recall is low - retrieval may be missing key information")
        if agg.get("faithfulness", 0) < 0.85:
            recommendations.append("🔴 Faithfulness is below threshold - model may be hallucinating")
        if agg.get("answer_relevancy", 0) < 0.80:
            recommendations.append("⚠️  Answer relevancy is low - responses may not address the question")

        if recommendations:
            for rec in recommendations:
                lines.append(f"- {rec}")
        else:
            lines.append("- ✅ All metrics meet target thresholds")
    else:
        lines.append("- Run incomplete - fix errors and re-run for full evaluation")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by Ragas evaluation runner at {run_payload.get('timestamp_utc', 'unknown')}*")

    return "\n".join(lines)


def print_console_summary(run_payload: Dict[str, Any]) -> None:
    """Print human-readable summary to console."""
    print("\n" + "=" * 80)
    print(f"  RAGAS EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nRun: {run_payload.get('run_id', 'unknown')}")
    print(f"Variant: {run_payload.get('variant', 'unknown')}")
    print(f"Status: {run_payload.get('status', 'unknown').upper()}")

    counts = run_payload.get("counts", {})
    print(f"\nExecution:")
    print(f"  Total: {counts.get('total', 0)} | Evaluated: {counts.get('evaluated', 0)} | Errors: {counts.get('errors', 0)}")

    metrics = run_payload.get("metrics", {})
    if metrics and metrics.get("aggregate"):
        print(f"\nMetrics:")
        agg = metrics.get("aggregate", {})
        for metric, value in sorted(agg.items()):
            formatted = format_metric_value(value, metric)
            print(f"  {metric:25s}: {formatted}")

    print("\n" + "=" * 80 + "\n")


def validate_config_schema(config: Dict[str, Any]) -> Optional[str]:
    """Validate config structure. Returns error message or None."""
    required = ["backend", "defaults", "variants", "paths", "reporting"]
    missing = [k for k in required if k not in config]
    if missing:
        return f"Missing required config keys: {missing}"

    variants = config.get("variants", {})
    if not variants:
        return "Config must have at least one variant defined"

    for name, variant in variants.items():
        if "payload_flags" not in variant:
            return f"Variant '{name}' missing 'payload_flags'"

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ragas evaluation runner (Amber scaffolding).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (see config.example.yaml).")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (jsonl or csv).")
    parser.add_argument("--variant", type=str, required=True, help="Variant key from config.variants.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path (reports/ragas/...).")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N requests (0 to disable).")
    parser.add_argument("--progress-interval", type=int, default=5, help="Print progress every N seconds (0 to disable).")
    args = parser.parse_args()

    config_path = Path(args.config)
    dataset_path = Path(args.dataset)
    out_path = Path(args.out)

    config = load_yaml_config(config_path)

    # Validate config schema
    validation_error = validate_config_schema(config)
    if validation_error:
        print(f"[ERROR] Invalid config: {validation_error}", file=sys.stderr)
        sys.exit(1)

    variant_cfg = resolve_variant(config, args.variant)

    defaults = config.get("defaults") or {}
    metric_names = (config.get("reporting") or {}).get("metrics") or []

    records = load_dataset(dataset_path)
    max_examples = defaults.get("max_examples")
    if max_examples:
        records = records[: int(max_examples)]

    # Pre-generate run info
    run_timestamp = datetime.now(timezone.utc)
    run_id = f"{args.variant}-{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}"

    # Setup OpenTelemetry instrumentation
    instrumentation_ctx = None
    try:
        from evals.ragas.otel_ragas import instrument_ragas_benchmark
        instrumentation_ctx = instrument_ragas_benchmark(
            run_id=run_id,
            variant=args.variant,
            num_samples=len(records),
            config=config
        )
    except Exception:
        import contextlib
        instrumentation_ctx = contextlib.nullcontext()

    # Call backend to get live answers/contexts (1:1 with user responses)
    with instrumentation_ctx:
        try:
            records = asyncio.run(
                fetch_backend_answers(
                    records,
                    config,
                    variant_cfg,
                    progress_every=args.progress_every,
                    progress_interval_seconds=args.progress_interval,
                )
            )
        except KeyboardInterrupt:
            print("[WARN] Evaluation interrupted by user; partial results (if any) not written.", file=sys.stderr)
            return

        errors = [r.get("__error") for r in records if r.get("__error")]
        prepared = prepare_samples(records)
        missing_answer = prepared["missing_answer"]
        missing_contexts = prepared["missing_contexts"]
        samples = prepared["samples"]

        aggregates: Optional[Dict[str, Any]] = None
        status = "pending"
        notes: List[str] = []

        error_count = len(errors)

        if error_count:
            status = "errors"
            notes.append(f"{error_count} request(s) failed; see records with '__error'.")
        if missing_answer:
            status = "incomplete"
            notes.append(f"{missing_answer} samples missing answers (wire backend call or prefill response).")
        if missing_contexts:
            notes.append(f"{missing_contexts} samples missing contexts (ensure retrieval outputs are captured).")

        if status in {"pending", "evaluated"}:
            status = "evaluated"
            aggregates = compute_ragas_metrics(samples, metric_names)
            if aggregates is None:
                status = "metrics_failed"
                notes.append("Metric computation skipped or failed (install optional deps or fix errors).")

    # Collate sample errors (up to 5) for quick visibility
    sample_errors = []
    for rec in records:
        err = rec.get("__error")
        if err and err not in sample_errors:
            sample_errors.append(err)
        if len(sample_errors) >= 5:
            break

    run_payload = {
        "run_id": run_id,
        "variant": args.variant,
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "git_commit": try_get_git_commit(),
        "timestamp_utc": run_timestamp.isoformat(),
        "variant_flags": variant_cfg.get("payload_flags") or {},
        "defaults": defaults,
        "counts": {
            "total": len(records),
            "evaluated": len(records) - (missing_answer + error_count),
            "missing_answer": missing_answer,
            "missing_contexts": missing_contexts,
            "errors": error_count,
        },
        "sample_errors": sample_errors,
        "metrics": aggregates,
        "status": status,
        "notes": notes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON for programmatic use
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)
    print(f"[OK] Wrote JSON report to {out_path}")

    # Write human-readable markdown summary
    md_path = out_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(format_run_summary_markdown(run_payload))
    print(f"[OK] Wrote markdown summary to {md_path}")

    # Print console summary
    print_console_summary(run_payload)

    if notes:
        for note in notes:
            print(f"[NOTE] {note}", file=sys.stderr)


if __name__ == "__main__":
    main()
