"""
RAGAS control API router.

Provides endpoints for:
- Running RAGAS benchmarks asynchronously
- Polling job status
- Listing available datasets and variants
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job storage (use Redis in production)
jobs: Dict[str, Dict[str, Any]] = {}


class BenchmarkRequest(BaseModel):
    dataset_path: str
    variants: List[str]
    config_overrides: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Dataset(BaseModel):
    path: str
    name: str
    sample_count: int


class Variant(BaseModel):
    key: str
    label: str
    description: str


async def run_benchmark_task(job_id: str, dataset_path: str, variants: List[str], config_overrides: Optional[Dict[str, Any]] = None):
    """
    Background task to run RAGAS benchmark.

    Executes the RAGAS evaluation script as a subprocess and updates job status.
    """
    import subprocess

    try:
        logger.info(f"Starting RAGAS benchmark job {job_id} for variants: {variants}")
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 0.0

        # Build command to run RAGAS benchmark
        # Using uv run to execute in the correct environment
        cmd = [
            "uv", "run", "python",
            "evals/ragas/run_ragas_benchmarks.py",
            "--dataset", dataset_path,
            "--variants", ",".join(variants),
        ]

        if config_overrides:
            # TODO: Support config overrides via CLI args or temp file
            pass

        # Execute benchmark
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"  # Docker container path
        )

        # Monitor progress (simplified - real implementation would parse output)
        while process.poll() is None:
            await asyncio.sleep(2)
            # Update progress based on elapsed time (rough estimate)
            jobs[job_id]["progress"] = min(jobs[job_id]["progress"] + 5, 95)

        # Check exit code
        if process.returncode == 0:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100.0

            # Load results from reports directory
            try:
                from evals.ragas.prometheus_exporter import RagasPrometheusExporter
                exporter = RagasPrometheusExporter()
                stats = exporter.get_json_stats()
                jobs[job_id]["results"] = stats
            except Exception as e:
                logger.warning(f"Could not load results: {e}")
                jobs[job_id]["results"] = {"status": "completed"}

            logger.info(f"RAGAS benchmark job {job_id} completed successfully")
        else:
            stderr = process.stderr.read() if process.stderr else "Unknown error"
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = stderr
            logger.error(f"RAGAS benchmark job {job_id} failed: {stderr}")

    except Exception as e:
        logger.error(f"Error in RAGAS benchmark job {job_id}: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@router.post("/ragas/run-benchmark")
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """
    Start a RAGAS benchmark run asynchronously.

    Args:
        request: Benchmark configuration (dataset, variants, config overrides)
        background_tasks: FastAPI background tasks

    Returns:
        Job ID and initial status
    """
    try:
        # Validate dataset exists
        dataset_path = Path(request.dataset_path)
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

        # Validate variants
        if not request.variants:
            raise HTTPException(status_code=400, detail="At least one variant must be specified")

        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "dataset": request.dataset_path,
            "variants": request.variants,
            "created_at": datetime.now().isoformat(),
        }

        # Start background task
        background_tasks.add_task(
            run_benchmark_task,
            job_id,
            request.dataset_path,
            request.variants,
            request.config_overrides
        )

        logger.info(f"Created RAGAS benchmark job {job_id}")

        return {"job_id": job_id, "status": "pending"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating RAGAS benchmark job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ragas/job/{job_id}")
async def get_job_status(job_id: str) -> JobResponse:
    """
    Get status of a RAGAS benchmark job.

    Args:
        job_id: Job identifier

    Returns:
        Job status, progress, and results (if completed)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return JobResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        results=job.get("results"),
        error=job.get("error"),
    )


@router.get("/ragas/datasets")
async def list_datasets() -> Dict[str, List[Dataset]]:
    """
    List available RAGAS test datasets.

    Returns:
        List of datasets with metadata (name, path, sample count)
    """
    try:
        datasets_dir = Path("evals/ragas/testsets")

        if not datasets_dir.exists():
            return {"datasets": []}

        datasets = []

        # Find all CSV files in testsets directory
        for csv_file in datasets_dir.glob("*.csv"):
            # Count lines to get sample count (minus header)
            with open(csv_file, 'r') as f:
                sample_count = sum(1 for _ in f) - 1  # Subtract header

            datasets.append(Dataset(
                path=str(csv_file),
                name=csv_file.stem.replace('_', ' ').title(),
                sample_count=max(0, sample_count)
            ))

        return {"datasets": datasets}

    except Exception as e:
        logger.error(f"Error listing datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ragas/variants")
async def list_variants() -> Dict[str, List[Variant]]:
    """
    List available RAGAS evaluation variants.

    Returns:
        List of variants with descriptions
    """
    try:
        # Read variants from config
        import yaml
        config_path = Path("evals/ragas/config.yaml")

        if not config_path.exists():
            # Return default variants
            return {
                "variants": [
                    Variant(
                        key="baseline",
                        label="Baseline",
                        description="Standard retrieval without graph enhancement"
                    ),
                    Variant(
                        key="graph_hybrid",
                        label="Graph Hybrid",
                        description="Hybrid retrieval combining vector search and graph traversal"
                    ),
                ]
            }

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        variants = []

        # Extract variants from config
        if "variants" in config:
            for variant_key, variant_config in config["variants"].items():
                variants.append(Variant(
                    key=variant_key,
                    label=variant_config.get("label", variant_key.replace('_', ' ').title()),
                    description=variant_config.get("description", f"{variant_key} retrieval variant")
                ))
        else:
            # Fallback to default variants
            variants = [
                Variant(
                    key="baseline",
                    label="Baseline",
                    description="Standard retrieval without graph enhancement"
                ),
                Variant(
                    key="graph_hybrid",
                    label="Graph Hybrid",
                    description="Hybrid retrieval combining vector search and graph traversal"
                ),
            ]

        return {"variants": variants}

    except Exception as e:
        logger.error(f"Error listing variants: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
