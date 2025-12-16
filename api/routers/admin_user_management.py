"""
Admin user management router for Phase 2.

Provides admin access to shared conversations, user activity logs, and metrics.
All endpoints require admin authentication (separate from user header auth).
"""

import json
import logging
import secrets
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query, Header, Request

from api.auth import require_admin_cookie
from api.auth import create_admin_session, invalidate_admin_session, _load_admin_token, validate_admin_session
from fastapi import Response, status, Cookie
from config.settings import settings
# Conversation sharing service removed; endpoints will return safe empty
# responses instead of importing the (now-unused) service to avoid import
# errors during startup.

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/response-config")
async def get_response_config():
    """
    Returns default response config for admin panel (stub).
    """
    # Example config, adjust as needed
    return {
        "max_response_length": 2048,
        "allow_mark_resolved": True,
        "show_conversation_snapshot": True,
        "fields": [
            "username", "reason", "message_count", "shared_at", "is_resolved", "resolved_notes"
        ]
    }
    
@router.get("/ping")
async def admin_ping(request: Request) -> dict:
    """
    Lightweight admin health/auth check.

    This endpoint is intentionally public so the frontend can call it before
    the operator has entered an admin token. It returns whether an admin token
    is configured on the server and whether the current request is authenticated
    (via `admin_session` cookie or `admin-token` header).
    """
    # Indicate whether admin token is configured on the server
    token_configured = bool(_load_admin_token())

    # Check for cookie session validity
    has_cookie = False
    session_valid = False
    sid = request.cookies.get("admin_session")
    if sid:
        has_cookie = True
        try:
            session_valid = bool(validate_admin_session(sid))
        except Exception:
            session_valid = False

    # Check for admin-token header presence and validity
    header_token = request.headers.get("admin-token") or request.headers.get("authorization")
    has_header = bool(header_token)
    header_valid = False
    if header_token:
        try:
            # if Authorization: Bearer <token> was used, verify_token will accept it
            from api.auth import verify_token

            # verify_token raises HTTPException on invalid token; treat that as not valid
            verify_token(header_token, require_admin=True)
            header_valid = True
        except Exception:
            header_valid = False

    return {
        "status": "ok",
        "admin_token_configured": token_configured,
        "has_cookie": has_cookie,
        "session_valid": session_valid,
        "has_header": has_header,
        "header_valid": header_valid,
    }


@router.get("/debug")
async def admin_debug(request: Request):
    """
    Dev helper: return booleans indicating whether the request included
    an `admin_session` cookie and/or an `admin-token` header and whether
    the session (if present) is valid. Safe for debugging (no secrets).
    """
    # Only enable in non-production to avoid exposure in prod
    try:
        if getattr(settings, "ENVIRONMENT", "dev") == "prod":
            raise HTTPException(status_code=404, detail="Not found")
    except Exception:
        # If settings missing ENVIRONMENT, allow by default (dev)
        pass

    has_cookie = False
    session_valid = False
    sid = request.cookies.get("admin_session")
    if sid:
        has_cookie = True
        try:
            session_valid = bool(validate_admin_session(sid))
        except Exception:
            session_valid = False

    has_header = bool(request.headers.get("admin-token"))

    return {"has_cookie": has_cookie, "session_valid": session_valid, "has_header": has_header}


@router.post("/login")
async def admin_login(payload: Dict[str, str], response: Response, request: Request):
    """
    Login using admin token as password. Sets an HttpOnly cookie `admin_session`.
    Request body: {"password": "<admin-token>"}
    """
    password = payload.get("password") if isinstance(payload, dict) else None
    client_host = None
    try:
        client_host = request.client.host if request.client else None
    except Exception:
        client_host = None
    # Mask token for logs: keep first 4 chars only
    masked_pw = None
    if password:
        try:
            masked_pw = (password[:4] + '...') if len(password) > 4 else '****'
        except Exception:
            masked_pw = '****'
    logger.debug("Admin login attempt from %s (password masked=%s)", client_host or 'unknown', masked_pw)
    stored = _load_admin_token()
    if not stored:
        raise HTTPException(status_code=500, detail="Admin token not configured")
    if not password or not secrets.compare_digest(password, stored):
        logger.debug("Admin login failed from %s (password masked=%s)", client_host or 'unknown', masked_pw)
        raise HTTPException(status_code=403, detail="Invalid credentials")

    sid = create_admin_session()
    # Set cookie. Set `secure` only when the incoming request scheme is HTTPS
    # so local dev over plain HTTP (localhost) still receives the cookie.
    secure_flag = request.url.scheme == "https"
    response.set_cookie(
        "admin_session",
        sid,
        httponly=True,
        samesite="lax",
        secure=secure_flag,
        max_age=int(getattr(__import__("api.auth"), "SESSION_TTL", 60 * 60 * 8)),
    )
    # Log cookie attributes and a masked session id for debugging (do not log secrets)
    try:
        masked_sid = (sid[:4] + '...') if sid and len(sid) > 4 else '****'
    except Exception:
        masked_sid = '****'
    logger.info(
        "Admin login succeeded from %s - setting admin_session cookie (sid masked=%s, secure=%s, samesite=%s, max_age=%s)",
        client_host or 'unknown',
        masked_sid,
        secure_flag,
        'lax',
        int(getattr(__import__("api.auth"), "SESSION_TTL", 60 * 60 * 8)),
    )
    return {"status": "ok"}


@router.post("/logout")
async def admin_logout(response: Response, admin_session: Optional[str] = Cookie(None)):
    """Invalidate admin session and clear cookie."""
    if admin_session:
        invalidate_admin_session(admin_session)
    response.delete_cookie("admin_session")
    return {"status": "ok"}


@router.get("/admin/dashboard-config", response_model=Dict[str, Any])
async def get_dashboard_config(
    admin_session: str = Depends(require_admin_cookie),
):
    """
    Get the dashboard configuration for the admin panel.

    Returns:
        dict: Configuration settings for the admin dashboard
    """
    # Example config, adjust as needed
    return {
        "widgets": [
            {"type": "shared_conversations", "title": "Recent Shared Conversations"},
            {"type": "user_activity", "title": "User Activity Overview"},
            {"type": "system_metrics", "title": "System Metrics"},
        ],
        "refresh_interval": 5,  # in minutes
    }


@router.get("/shared-conversations")
async def list_shared_conversations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    username: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    admin_session: str = Depends(require_admin_cookie),
):
    """
    List all shared conversations (admin endpoint).

    Query Parameters:
        - limit: Max results (1-200, default 50)
        - offset: Pagination offset
        - username: Filter by sharing user
        - start_date: Filter by start date (ISO format: 2025-12-04)
        - end_date: Filter by end date (ISO format: 2025-12-04)

    Returns:
        List of shared conversations with metadata
    """
    # Conversation sharing feature removed; return an empty paginated response
    logger.info("Conversation sharing disabled - returning empty result for admin list_shared_conversations")
    return {
        "conversations": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/shared-conversations/{shared_id}")
async def get_shared_conversation(
    shared_id: str,
    admin_session: str = Depends(require_admin_cookie),
):
    """
    Retrieve a specific shared conversation (admin endpoint).

    Path Parameters:
        - shared_id: Unique share ID

    Returns:
        Full shared conversation including snapshot with all message fields
    """
    # Conversation sharing removed; indicate the endpoint is no longer available.
    logger.info("Conversation sharing disabled - get_shared_conversation called for %s", shared_id)
    raise HTTPException(status_code=501, detail="Conversation sharing feature is disabled")


@router.post("/shared-conversations/{shared_id}/mark-resolved")
async def mark_conversation_resolved(
    shared_id: str,
    notes: Optional[str] = Query(None),
    admin_session: str = Depends(require_admin_cookie),
):
    """
    Mark a shared conversation as resolved (admin endpoint).

    Path Parameters:
        - shared_id: Unique share ID

    Query Parameters:
        - notes: Optional resolution notes/action taken

    Returns:
        Updated shared conversation record
    """
    logger.info("Conversation sharing disabled - mark_conversation_resolved called for %s", shared_id)
    raise HTTPException(status_code=501, detail="Conversation sharing feature is disabled")


@router.get("/shared-conversations-stats")
async def get_sharing_stats(
    admin_session: str = Depends(require_admin_cookie),
):
    """
    Get sharing statistics (admin endpoint).

    Returns:
        Stats including total shared, by user, by date
    """
    # Conversation sharing removed; return empty stats
    logger.info("Conversation sharing disabled - returning empty stats")
    return {
        "total_shares": 0,
        "shares_by_user": [],
        "shares_over_time": [],
    }


@router.get("/user-activity")
async def get_user_activity(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    activity_type: Optional[str] = Query(None),
    username: Optional[str] = Query(None),
    admin_session: str = Depends(require_admin_cookie),
):
    """
    Get raw user activity logs (admin endpoint).

    Query Parameters:
        - limit: Max results (1-500, default 100)
        - offset: Pagination offset
        - activity_type: Filter by type (query, conversation_share, session_delete, etc)
        - username: Filter by username

    Returns:
        Paginated activity log entries
    """
    try:
        from api.user_panel.services.user_activity_service import (
            get_user_activity_service,
        )
        from pathlib import Path

        activity_service = get_user_activity_service()

        # Read JSONL file and parse entries
        log_file = Path(activity_service.log_path)

        if not log_file.exists():
            return {
                "activities": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
            }

        # Read all entries (for simplicity, can be optimized for large files)
        all_entries = []
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        all_entries.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading activity log: {e}")
            raise HTTPException(status_code=500, detail="Error reading activity logs")

        # Aggregate data per user
        user_activity_stats = {}
        for entry in all_entries:
            username = entry.get("username")
            if not username:
                continue

            if username not in user_activity_stats:
                user_activity_stats[username] = {
                    "username": username,
                    "query_count": 0,
                    "share_count": 0,
                    "last_query_at": None,
                    "latencies": [],
                }

            if entry.get("type") == "query":
                user_activity_stats[username]["query_count"] += 1
                if entry.get("latency_ms") is not None:
                    user_activity_stats[username]["latencies"].append(entry["latency_ms"])
                if (
                    entry.get("timestamp")
                    and (user_activity_stats[username]["last_query_at"] is None
                    or entry["timestamp"] > user_activity_stats[username]["last_query_at"])
                ):
                    user_activity_stats[username]["last_query_at"] = entry["timestamp"]
            elif entry.get("type") == "share":
                user_activity_stats[username]["share_count"] += 1

        # Finalize user activity stats
        final_user_activities = []
        for username, stats in user_activity_stats.items():
            avg_latency = (
                statistics.mean(stats["latencies"]) if stats["latencies"] else 0
            )
            final_user_activities.append({
                "username": username,
                "query_count": stats["query_count"],
                "share_count": stats["share_count"],
                "last_query_at": stats["last_query_at"] if stats["last_query_at"] else "",
                "avg_latency_ms": round(avg_latency, 2),
            })

        # Sort by last_query_at descending
        final_user_activities.sort(
            key=lambda x: x.get("last_query_at", ""), reverse=True
        )

        total = len(final_user_activities)
        paginated = final_user_activities[offset : offset + limit]

        logger.info(
            f"Admin viewed user activity (limit={limit}, offset={offset}, type={activity_type}, user={username})"
        )

        return {
            "activities": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user activity: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving activity logs")


@router.get("/user-activity-metrics")
async def get_user_activity_metrics(
    days: int = Query(7, ge=1, le=90),
    admin_session: str = Depends(require_admin_cookie),
):
    """
    Get aggregated user activity metrics (admin endpoint).

    Query Parameters:
        - days: Number of days to aggregate (1-90, default 7)

    Returns:
        Metrics: total_queries, active_users, avg_latency, most_active_users
    """
    try:
        from api.user_panel.services.user_activity_service import (
            get_user_activity_service,
        )
        from pathlib import Path
        import statistics

        activity_service = get_user_activity_service()
        log_file = Path(activity_service.log_path)

        if not log_file.exists():
            return {
                "total_queries": 0,
                "active_users": 0,
                "avg_latency_ms": 0,
                "most_active_users": [],
                "query_distribution": {},
                "error_count": 0,
            }

        # Read and parse entries
        all_entries = []
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        # Filter by date
                        if entry.get("timestamp"):
                            entry_date = datetime.fromisoformat(entry["timestamp"])
                            if entry_date >= cutoff_date:
                                all_entries.append(entry)
        except Exception as e:
            logger.error(f"Error reading activity log: {e}")

        # Calculate metrics
        query_entries = [e for e in all_entries if e.get("type") == "query"]
        share_entries = [e for e in all_entries if e.get("type") == "conversation_share"]
        error_entries = [e for e in query_entries if e.get("error")]

        total_queries = len(query_entries)
        total_shares = len(share_entries)

        # Total unique users (within the `days` period)
        total_users_set = set(e.get("username") for e in all_entries if e.get("username"))
        total_users_count = len(total_users_set)

        # Average and P95 latency
        latencies = [e.get("latency_ms", 0) for e in query_entries if e.get("latency_ms")]
        avg_latency = (
            statistics.mean(latencies) if latencies else 0
        )
        p95_latency = (
            statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 100 else (max(latencies) if latencies else 0)
        )

        # Metrics for last 24 hours
        last_24h_cutoff = datetime.utcnow() - timedelta(days=1)
        entries_last_24h = [
            e for e in all_entries
            if e.get("timestamp") and datetime.fromisoformat(e["timestamp"]) >= last_24h_cutoff
        ]
        active_users_today_set = set(e.get("username") for e in entries_last_24h if e.get("username"))
        active_users_today_count = len(active_users_today_set)
        queries_last_24h_count = len([e for e in entries_last_24h if e.get("type") == "query"])

        # Most active users
        user_counts = {}
        for entry in all_entries:
            if entry.get("username"):
                user_counts[entry["username"]] = user_counts.get(entry["username"], 0) + 1

        most_active_users = sorted(
            user_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Query distribution by type
        query_types = {}
        for entry in all_entries:
            qtype = entry.get("type", "unknown")
            query_types[qtype] = query_types.get(qtype, 0) + 1

        logger.info(f"Admin viewed activity metrics (days={days})")

        return {
            "total_queries": total_queries,
            "total_shares": total_shares,
            "total_users": total_users_count,
            "active_users": active_users_count,
            "active_users_today": active_users_today_count,
            "queries_last_24h": queries_last_24h_count,
            "avg_query_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "error_count": len(error_entries),
            "error_rate": round(len(error_entries) / max(total_queries, 1), 4),
            "most_active_users": [
                {"username": user, "activity_count": count}
                for user, count in most_active_users
            ],
            "activity_distribution": query_types,
            "time_period_days": days,
        }

    except Exception as e:
        logger.error(f"Error calculating user activity metrics: {e}")
        raise HTTPException(status_code=500, detail="Error calculating metrics")


@router.get("/users")
async def list_users(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    admin_session: str = Depends(require_admin_cookie),
):
    """
    Get list of all users who have used the system (admin endpoint).

    Query Parameters:
        - limit: Max results (1-500, default 100)
        - offset: Pagination offset

    Returns:
        List of unique users with activity counts
    """
    try:
        from api.user_panel.services.user_activity_service import (
            get_user_activity_service,
        )
        from pathlib import Path

        activity_service = get_user_activity_service()
        log_file = Path(activity_service.log_path)

        if not log_file.exists():
            return {"users": [], "total": 0, "limit": limit, "offset": offset}

        # Read and parse entries
        user_stats = {}

        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        username = entry.get("username")
                        if username:
                            if username not in user_stats:
                                user_stats[username] = {
                                    "username": username,
                                    "activity_count": 0,
                                    "query_count": 0,
                                    "share_count": 0,
                                    "last_active": entry.get("timestamp"),
                                }
                            user_stats[username]["activity_count"] += 1
                            if entry.get("type") == "query":
                                user_stats[username]["query_count"] += 1
                            elif entry.get("type") == "conversation_share":
                                user_stats[username]["share_count"] += 1
                            # Update last active time
                            if entry.get("timestamp") > user_stats[username]["last_active"]:
                                user_stats[username]["last_active"] = entry.get("timestamp")
        except Exception as e:
            logger.error(f"Error reading activity log: {e}")

        # Aggregate data per user
        user_profiles = []
        for username, stats in user_stats.items():
            user_profiles.append({
                "username": username,
                "query_count": stats["query_count"],
                "share_count": stats["share_count"],
                "last_active": stats["last_active"],
            })

        # Sort by last active descending
        user_profiles.sort(
            key=lambda x: x.get("last_active", ""), reverse=True
        )

        total = len(user_profiles)
        paginated = user_profiles[offset : offset + limit]

        logger.info(f"Admin listed users (limit={limit}, offset={offset})")

        return {
            "users": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving user list")
