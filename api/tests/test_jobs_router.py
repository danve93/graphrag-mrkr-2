from fastapi.testclient import TestClient
from api.main import app
from api import job_manager
import os

# ensure admin token is present for tests
os.environ["JOBS_ADMIN_TOKEN"] = os.environ.get("JOBS_ADMIN_TOKEN", "admintoken")

client = TestClient(app)


def test_jobs_list_and_control():
    # create a couple of jobs in the configured manager
    job_manager.create_job("job-alpha", message="first")
    job_manager.create_job("job-beta", message="second")
    # create several jobs to test pagination
    for i in range(1, 8):
        job_manager.create_job(f"job-{i}", message=f"msg-{i}")

    headers = {"Authorization": "Bearer admintoken"}

    # list jobs
    r = client.get("/api/jobs", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert "jobs" in data
    ids = {j["job_id"] for j in data["jobs"]}
    assert "job-alpha" in ids
    assert "job-beta" in ids

    # pause
    r = client.post("/api/jobs/pause", headers=headers)
    assert r.status_code == 200
    assert r.json().get("status") == "paused"
    assert job_manager.is_paused() is True

    # resume
    r = client.post("/api/jobs/resume", headers=headers)
    assert r.status_code == 200
    assert r.json().get("status") == "running"
    assert job_manager.is_paused() is False

    # requeue (in-memory returns 0)
    r = client.post("/api/jobs/requeue", headers=headers)
    assert r.status_code == 200
    assert "requeued" in r.json()


def test_job_details_and_cancel_and_pagination():
    # ensure admin token is set for admin endpoints
    os.environ["JOBS_ADMIN_TOKEN"] = "admintoken"
    # ensure job exists
    job_manager.create_job("detail-job", message="detail")
    r = client.get("/api/jobs/detail-job", headers={"Authorization": "Bearer admintoken"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("job_id") == "detail-job"

    # admin endpoint to retrieve persistent user token
    r = client.get("/api/jobs/token", headers={"Authorization": "Bearer admintoken"})
    assert r.status_code == 200
    tok = r.json().get("user_token")
    assert isinstance(tok, str) and len(tok) > 0

    # cancel the job
    r = client.post("/api/jobs/detail-job/cancel", headers={"Authorization": "Bearer admintoken"})
    assert r.status_code == 200
    assert r.json().get("status") == "cancelled"
    job = job_manager.get_job("detail-job")
    assert job and job.get("status") == "cancelled"

    # test pagination: page size 3
    r = client.get("/api/jobs?page=1&page_size=3", headers={"Authorization": "Bearer admintoken"})
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("page") == 1
    assert payload.get("page_size") == 3
    assert "total" in payload
    assert isinstance(payload.get("jobs"), list)


def test_admin_cancel_all_purge_and_retry():
    admin_headers = {"Authorization": "Bearer admintoken"}
    # ensure a few jobs exist
    for i in range(3):
        job_manager.create_job(f"admin-job-{i}", message=f"m{i}")

    # cancel-all
    r = client.post("/api/jobs/cancel-all", headers=admin_headers)
    assert r.status_code == 200
    cancelled = r.json().get("cancelled")
    assert isinstance(cancelled, int) and cancelled >= 1

    # create jobs again to test purge
    for i in range(2):
        job_manager.create_job(f"purge-job-{i}", message=f"p{i}")

    r = client.post("/api/jobs/purge", headers=admin_headers)
    assert r.status_code == 200
    removed = r.json().get("removed")
    assert isinstance(removed, int) and removed >= 1

    # test retry with user token
    job_manager.create_job("to-retry", message="needs retry")
    job_manager.set_job_failed("to-retry", "failed for test")
    # get the persistent user token (admin-only endpoint)
    r = client.get("/api/jobs/token", headers=admin_headers)
    assert r.status_code == 200
    user_token = r.json().get("user_token")
    assert user_token

    # user retries the job
    r = client.post("/api/jobs/to-retry/retry", headers={"Authorization": f"Bearer {user_token}"})
    assert r.status_code == 200
    assert r.json().get("status") == "requeued"
    j = job_manager.get_job("to-retry")
    assert j and j.get("status") in ("queued", "running", "requeued", "success")

