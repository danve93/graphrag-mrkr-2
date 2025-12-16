import requests
import time
import os

BASE_URL = "http://127.0.0.1:8000/api"

def print_result(step, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {step}: {details}")

def main():
    print("Starting Verification...")
    
    # 1. Create Admin User
    # We first identify as a normal user
    print("\n--- 1. Bootstrap Admin ---")
    resp = requests.post(f"{BASE_URL}/users/identify", json={"username": "admin_tester"})
    if resp.status_code != 200:
        print_result("Identify Admin", False, resp.text)
        return
    
    admin_data = resp.json()
    admin_id = admin_data["user_id"]
    admin_token = admin_data["token"]
    print(f"Created user {admin_id}. Role: {admin_data.get('role')}")
    
    # Manually elevate to admin (Simulating DB access)
    # Since I cannot easily execute cypher from here without driver, 
    # I will assume I can't verify 'role=admin' logic fully strictly via API without backend helper 
    # BUT I can rely on the fact that I can edit the DB. 
    # Wait, I have access to `graph_db` in the environment if I run this INSIDE backend container 
    # OR I can use the `database` router if exposed? No.
    
    # I will rely on the `graph_db` exposed via `neo4j` package if I ran detailed script, 
    # but here I will use `curl` or `run_command` in the agent to upgrade the user.
    # So I will pause this script, run the upgrade command, then continue?
    # No, I can't pause. I will just run the command using `os.system` via `docker exec`?
    # Yes, since I am on the host.
    
    cmd = f'docker exec graphrag-backend python3 -c "from core.graph_db import graph_db; graph_db.driver.execute_query(\'MATCH (u:User {{id: \\"{admin_id}\\"}}) SET u.role = \\"admin\\"\')"'
    os.system(cmd)
    
    # Verify Admin Role
    resp = requests.post(f"{BASE_URL}/users/identify", json={"username": "admin_tester"})
    admin_data = resp.json()
    admin_token = admin_data["token"] # Update token!
    if admin_data.get("role") == "admin":
        print_result("Elevate to Admin", True)
    else:
        print_result("Elevate to Admin", False, f"Role is {admin_data.get('role')}")
        return
        
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    
    # 2. Admin Creates API Key
    print("\n--- 2. API Key Management ---")
    key_payload = {"name": "Test Key", "role": "external"}
    resp = requests.post(f"{BASE_URL}/admin/api-keys", json=key_payload, headers=admin_headers)
    if resp.status_code == 200:
        key_data = resp.json()
        api_key = key_data["key"]
        print_result("Create API Key", True, f"Key: {api_key[:5]}...")
    else:
        print_result("Create API Key", False, resp.text)
        return

    # 3. External User Login
    print("\n--- 3. External User Flow ---")
    ext_payload = {"username": "external_user_1", "api_key": api_key}
    resp = requests.post(f"{BASE_URL}/users/identify", json=ext_payload)
    if resp.status_code == 200:
        ext_data = resp.json()
        ext_token = ext_data["token"]
        ext_id = ext_data["user_id"]
        role = ext_data.get("role")
        if role == "external":
            print_result("Identify with API Key", True, f"Role: {role}")
        else:
             print_result("Identify with API Key", False, f"Wrong Role: {role}")
    else:
        print_result("Identify with API Key", False, resp.text)
        return

    ext_headers = {"Authorization": f"Bearer {ext_token}"}

    # 4. Chat and Share
    print("\n--- 4. Chat & Share ---")
    # Create session
    # Note: query endpoint usually creates session if not provided, but we want session_id
    # We can use /history/sessions to create or just send a msg.
    # Let's send a simple msg.
    chat_payload = {"message": "Hello from external", "stream": False}
    resp = requests.post(f"{BASE_URL}/chat/query", json=chat_payload, headers=ext_headers)
    if resp.status_code == 200:
        chat_resp = resp.json()
        session_id = chat_resp.get("session_id")
        print_result("Send Message", True, f"Session: {session_id}")
    else:
        print_result("Send Message", False, resp.text)
        return

    # Share Session
    resp = requests.post(f"{BASE_URL}/history/{session_id}/share", headers=ext_headers)
    if resp.status_code == 200:
        print_result("Share Session", True)
    else:
        print_result("Share Session", False, resp.text)
        return

    # 5. Admin Views Shared Session
    print("\n--- 5. Admin View ---")
    # List shared
    resp = requests.get(f"{BASE_URL}/history/admin/shared", headers=admin_headers)
    if resp.status_code == 200:
        shared = resp.json()
        found = any(s["session_id"] == session_id for s in shared)
        if found:
            print_result("List Shared Sessions", True, "Session found in list")
        else:
            print_result("List Shared Sessions", False, "Session NOT found in list")
    else:
        print_result("List Shared Sessions", False, resp.text)

    # Get conversation details as Admin
    # Explicitly check if Admin can read this user's session
    resp = requests.get(f"{BASE_URL}/history/{session_id}", headers=admin_headers)
    if resp.status_code == 200:
        msgs = resp.json().get("messages", [])
        if len(msgs) > 0 and msgs[0]["content"] == "Hello from external":
            print_result("Admin Read Shared Session", True, "Content matched")
        else:
            print_result("Admin Read Shared Session", False, "Content mismatch or empty")
    else:
        print_result("Admin Read Shared Session", False, resp.text)
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
