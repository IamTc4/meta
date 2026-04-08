import requests
import os
import json

# Your HF Space Direct API URL
HF_API_URL = "https://developerbee-social-graph-env.hf.space"
TOKEN = os.getenv("HF_TOKEN", "")

def test_ping():
    print(f"Pinging HF Space at {HF_API_URL}...")
    try:
        # 1. Test /reset
        print("Testing /reset...")
        resp = requests.post(f"{HF_API_URL}/reset", headers={"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})
        if resp.status_code == 200:
            print("✅ Reset Successful!")
            obs = resp.json()
            print(f"Initial Observation: {json.dumps(obs, indent=2)[:200]}...")
            
            # 2. Test /state
            print("\nTesting /state...")
            state_resp = requests.get(f"{HF_API_URL}/state")
            if state_resp.status_code == 200:
                 print(f"✅ State retrieved: {state_resp.json()}")
            
            # 3. Test /step
            print("\nTesting /step...")
            # Sample FLAG_ACCOUNT action
            action = {
                "action_type": "FLAG_ACCOUNT",
                "target_ids": ["test_id"],
                "confidence": 0.5,
                "reasoning": "testing remote server"
            }
            step_resp = requests.post(f"{HF_API_URL}/step", json={"action": action})
            if step_resp.status_code == 200:
                print("✅ Step Successful!")
                print(f"Step Result: {json.dumps(step_resp.json(), indent=2)[:200]}...")
            else:
                print(f"❌ Step Failed: {step_resp.status_code} - {step_resp.text}")
        else:
            print(f"❌ Reset Failed: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")

if __name__ == "__main__":
    test_ping()
