"""
Aggregate Ragas run outputs and emit a Markdown summary.

Inputs: one or more JSON files produced by `ragas_runner.py`.
Optional: baseline JSON to enforce regression thresholds (-5%).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_run(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        data["_path"] = str(path)
        return data


def collect_metric_names(runs: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for run in runs:
        metrics = (run.get("metrics") or {}).get("aggregate") or {}
        names.update(metrics.keys())
    return sorted(names)


def format_val(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def regression_check(current: Dict[str, float], baseline: Dict[str, float], threshold: float = -0.05) -> List[str]:
    findings: List[str] = []
    for metric, base_val in baseline.items():
        if base_val is None or metric not in current:
            continue
        delta = (current[metric] - base_val) / base_val if base_val else 0.0
        if delta < threshold:
            pct = delta * 100.0
            findings.append(f"{metric}: {pct:.1f}% vs baseline")
    return findings


def build_markdown(
    runs: List[Dict[str, Any]],
    metric_names: List[str],
    baseline: Optional[Dict[str, float]],
    regressions: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Ragas Evaluation Summary")
    lines.append(f"_Generated_: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Runs")
    for run in runs:
        lines.append(
            f"- **{run.get('variant')}** | status: {run.get('status')} | "
            f"total: {run.get('counts', {}).get('total')} | "
            f"evaluated: {run.get('counts', {}).get('evaluated')} | "
            f"file: `{run.get('_path')}`"
        )
    lines.append("")

    lines.append("## Metrics")
    header = ["metric"] + [run.get("variant", "?") for run in runs]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for metric in metric_names:
        row = [metric]
        for run in runs:
            agg = (run.get("metrics") or {}).get("aggregate") or {}
            row.append(format_val(agg.get(metric)))
        lines.append("| " + " | ".join(row) + " |")

    if baseline:
        lines.append("")
        lines.append("## Baseline")
        lines.append("| metric | baseline |")
        lines.append("| --- | --- |")
        for metric, val in baseline.items():
            lines.append(f"| {metric} | {format_val(val)} |")

    if regressions:
        lines.append("")
        lines.append("## Regression Alerts (>-5% drop)")
        for item in regressions:
            lines.append(f"- {item}")
    else:
        lines.append("")
        lines.append("## Regression Alerts")
        lines.append("- None")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Cross-doc leakage and intent checks are not computed here; add doc-id filters in runner outputs to enable.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Ragas run outputs into Markdown.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of run JSON files.")
    parser.add_argument("--baseline", type=str, help="Optional baseline JSON file for regression checks.")
    parser.add_argument("--out", type=str, required=True, help="Markdown output path.")
    args = parser.parse_args()

    run_paths = [Path(p) for p in args.inputs]
    runs = [load_run(p) for p in run_paths]

    metric_names = collect_metric_names(runs)

    baseline_metrics: Optional[Dict[str, float]] = None
    regressions: List[str] = []
    if args.baseline:
        baseline_run = load_run(Path(args.baseline))
        baseline_metrics = (baseline_run.get("metrics") or {}).get("aggregate") or {}

        # Compare only the first non-baseline run against baseline
        if baseline_metrics and runs:
            current = (runs[0].get("metrics") or {}).get("aggregate") or {}
            regressions = regression_check(current, baseline_metrics)

    markdown = build_markdown(runs, metric_names, baseline_metrics, regressions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"[OK] Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
