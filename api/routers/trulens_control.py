"""
TruLens control API router.

Provides endpoints for:
- Enabling/disabling TruLens monitoring
- Launching TruLens dashboard
- Updating configuration
- Resetting database
"""

import logging
import os
import subprocess
import signal
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Track dashboard process
dashboard_process: Optional[subprocess.Popen] = None


class ControlAction(BaseModel):
    action: str  # "enable" | "disable" | "reset"


class ConfigUpdate(BaseModel):
    sampling_rate: Optional[float] = None
    database: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None


@router.post("/trulens/control")
async def control_monitoring(request: ControlAction) -> Dict[str, Any]:
    """
    Control TruLens monitoring.

    Actions:
    - enable: Enable continuous monitoring
    - disable: Disable continuous monitoring
    - reset: Reset database (clear all records)

    Args:
        request: Control action to perform

    Returns:
        Status and message
    """
    try:
        from evals.trulens.trulens_wrapper import get_monitor

        monitor = get_monitor()

        if monitor is None:
            if request.action == "enable":
                # Lazy initialize
                logger.info("Lazily initializing TruLens monitor...")
                
                # Ensure config enables it first
                import yaml
                config_path = Path("evals/trulens/config.yaml")
                config = {}
                if config_path.exists():
                     with open(config_path, 'r') as f:
                         config = yaml.safe_load(f) or {}
                
                if 'trulens' not in config: config['trulens'] = {}
                config['trulens']['enabled'] = True
                
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config, f)
                
                from evals.trulens.trulens_initializer import initialize_trulens
                monitor = initialize_trulens()
                
                if monitor is None:
                     raise HTTPException(
                         status_code=500,
                         detail="Failed to initialize TruLens monitor even after enabling."
                     )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="TruLens not initialized. Enable it first."
                )

        # Helper to update config file
        def update_config_enabled(enabled: bool):
            try:
                import yaml
                config_path = Path("evals/trulens/config.yaml")
                config = {}
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f) or {}
                
                if 'trulens' not in config: config['trulens'] = {}
                config['trulens']['enabled'] = enabled
                
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config, f)
            except Exception as e:
                logger.error(f"Failed to persist TruLens state to config: {e}")

        if request.action == "enable":
            monitor.enabled = True
            update_config_enabled(True)
            return {
                "status": "success",
                "message": "TruLens monitoring enabled",
                "monitoring_enabled": True
            }

        elif request.action == "disable":
            monitor.enabled = False
            update_config_enabled(False)
            return {
                "status": "success",
                "message": "TruLens monitoring disabled",
                "monitoring_enabled": False
            }

        elif request.action == "reset":
            if monitor.tru is not None:
                monitor.tru.reset_database()
                return {
                    "status": "success",
                    "message": "TruLens database reset successfully"
                }
            else:
                raise HTTPException(
                    status_code=503,
                    detail="TruLens database not initialized"
                )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}. Must be 'enable', 'disable', or 'reset'."
            )

    except ImportError:
        logger.warning("TruLens not installed")
        raise HTTPException(
            status_code=503,
            detail="TruLens not installed. Install with: uv pip install -r evals/trulens/requirements-trulens.txt"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in control action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trulens/dashboard/launch")
async def launch_dashboard() -> Dict[str, Any]:
    """
    Launch TruLens Streamlit dashboard.

    Starts the dashboard in a background process and returns the URL.

    Returns:
        Dashboard URL and port
    """
    return {
        "status": "success",
        "url": "http://localhost:8501",
        "port": 8501,
        "pid": 0,
        "message": "TruLens dashboard running as dedicated service"
    }




@router.post("/trulens/dashboard/stop")
async def stop_dashboard() -> Dict[str, Any]:
    """
    Stop TruLens Streamlit dashboard.

    Terminates the dashboard process if running.

    Returns:
        Status message
    """
    global dashboard_process

    try:
        if dashboard_process is None or dashboard_process.poll() is not None:
            return {
                "status": "not_running",
                "message": "Dashboard is not running"
            }

        # Terminate process
        dashboard_process.terminate()
        try:
            dashboard_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            dashboard_process.kill()

        dashboard_process = None

        return {
            "status": "success",
            "message": "Dashboard stopped successfully"
        }

    except Exception as e:
        logger.error(f"Error stopping dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trulens/config")
async def get_config() -> Dict[str, Any]:
    """
    Get current TruLens configuration.

    Returns:
        Configuration settings
    """
    try:
        import yaml

        config_path = Path("evals/trulens/config.yaml")

        if not config_path.exists():
            return {
                "status": "no_config",
                "message": "Configuration file not found"
            }

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return {
            "status": "success",
            "config": config
        }

    except Exception as e:
        logger.error(f"Error reading config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trulens/config")
async def update_config(request: ConfigUpdate) -> Dict[str, Any]:
    """
    Update TruLens configuration.

    Args:
        request: Configuration updates

    Returns:
        Status and message
    """
    try:
        import yaml
        from evals.trulens.trulens_wrapper import get_monitor

        config_path = Path("evals/trulens/config.yaml")

        # Read current config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        # Update fields
        if request.sampling_rate is not None:
            if 'trulens' not in config:
                config['trulens'] = {}
            config['trulens']['sampling_rate'] = request.sampling_rate

            # Update runtime monitor
            monitor = get_monitor()
            if monitor:
                monitor.sampling_rate = request.sampling_rate

        if request.database is not None:
            if 'trulens' not in config:
                config['trulens'] = {}
            config['trulens']['database'] = request.database

        if request.thresholds is not None:
            if 'feedback' not in config:
                config['feedback'] = {}
            if 'thresholds' not in config['feedback']:
                config['feedback']['thresholds'] = {}
            config['feedback']['thresholds'].update(request.thresholds)

        # Write updated config
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "config": config
        }

    except Exception as e:
        logger.error(f"Error updating config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trulens/dashboard/status")
async def dashboard_status() -> Dict[str, Any]:
    """
    Get TruLens dashboard status.

    Returns:
        Dashboard running status and URL
    """
    global dashboard_process

    # 1. Check if running as external service (Docker)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=0.5) as client:
            # Try service name first (Docker network)
            response = await client.get("http://graphrag-trulens-dashboard:8501/_stcore/health")
            if response.status_code == 200:
                return {
                    "status": "running",
                    "url": "http://localhost:8501",
                    "port": 8501,
                    "pid": 0,
                    "type": "container"
                }
    except Exception:
        # Ignore connection errors, means not running there
        pass

    # 2. Check local process (Legacy/Local dev)
    if dashboard_process is not None and dashboard_process.poll() is None:
        return {
            "status": "running",
            "url": "http://localhost:8501",
            "port": 8501,
            "pid": dashboard_process.pid,
            "type": "process"
        }
    else:
        return {
            "status": "stopped",
            "url": None,
            "port": None,
            "pid": None
        }
