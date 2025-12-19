
import requests
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

def reproduce_500():
    doc_id = 'bb374326496b924f7eb9cc37d121187e'
    print(f"Fetching document {doc_id}...")
    
    # We need the admin token? Authentication might be required.
    # The user was using the UI, so they were authenticated.
    # Let's try regular GET. If auth fails, we'll see 401/403.
    # If 500, then auth passed (or failed badly).
    
    # But wait, we restored the admin token.
    token = os.environ.get("JOBS_ADMIN_TOKEN")
    if not token:
        # try loading from .env
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("JOBS_ADMIN_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
        except: pass
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        # Also need X-API-Key? The backend accepts Bearer token which maps to API Key check.
    
    try:
        url = f"http://localhost:8000/api/documents/{doc_id}"
        print(f"Request: {url}")
        resp = requests.get(url, headers=headers)
        print(f"Status: {resp.status_code}")
        try:
            print(f"Body: {resp.json()}")
        except:
            print(f"Body: {resp.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    reproduce_500()
