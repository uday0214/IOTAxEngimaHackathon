"""
Verification script for server.py
Uses in-process threading to comply with Windows sandbox networking rules.
Tests status, experiments, holdout results, and parameter comparisons.
"""

import urllib.request
import json
import threading
import time
import sys
import os

from server import manager, RequestHandler, ReusableTCPServer, HOST

PORT = 8899
BASE_URL = f"http://{HOST}:{PORT}"

print(f"Starting in-process test server on {BASE_URL}...")
httpd = ReusableTCPServer((HOST, PORT), RequestHandler)
thread = threading.Thread(target=httpd.serve_forever, daemon=True)
thread.start()
time.sleep(0.5)

try:
    # 1. Test /api/status
    with urllib.request.urlopen(f"{BASE_URL}/api/status") as response:
        assert response.status == 200
        status_data = json.loads(response.read().decode())
        print("[Pass] /api/status -> Champion Run #", status_data["champion"]["run_id"])
        assert status_data["runs_completed"] == 15

    # 2. Test /api/experiments
    with urllib.request.urlopen(f"{BASE_URL}/api/experiments") as response:
        assert response.status == 200
        exp_data = json.loads(response.read().decode())
        print(f"[Pass] /api/experiments -> {len(exp_data['experiments'])} experiments returned")
        assert len(exp_data["experiments"]) == 15

    # 3. Test /api/holdout
    with urllib.request.urlopen(f"{BASE_URL}/api/holdout") as response:
        assert response.status == 200
        holdout = json.loads(response.read().decode())
        print(f"[Pass] /api/holdout -> Samples: {holdout['holdout_samples']}, ROC-AUC: {holdout['holdout_roc_auc']}")
        assert holdout["holdout_samples"] == 1409

    # 4. Test /api/parameters
    with urllib.request.urlopen(f"{BASE_URL}/api/parameters") as response:
        assert response.status == 200
        params = json.loads(response.read().decode())
        print("[Pass] /api/parameters -> Sensitivities keys:", list(params["sensitivities"].keys()))

    # 5. Test /api/llm-consult
    req_llm = urllib.request.Request(
        f"{BASE_URL}/api/llm-consult",
        data=json.dumps({"query": "Explain tree depth comparison"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_llm) as response:
        assert response.status == 200
        consult = json.loads(response.read().decode())
        print("[Pass] /api/llm-consult -> Reply snippet:", consult["reply"][:80], "...")

    # 6. Test GET /
    with urllib.request.urlopen(f"{BASE_URL}/") as response:
        assert response.status == 200
        html = response.read().decode()
        print(f"[Pass] GET / -> HTML size: {len(html)} bytes")

    print("\nALL SERVER TESTS PASSED PERFECTLY!")

finally:
    httpd.shutdown()
    httpd.server_close()
    print("Test server cleanly stopped.")
