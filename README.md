---
title: Social Graph Env
emoji: 🕵️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
---

# Social Graph Manipulation Detection Environment

> **OpenEnv-compliant** · Meta PyTorch Hackathon submission

An agent is dropped into a synthetic social network and must identify
**Coordinated Inauthentic Behavior (CIB)** — bot farms, astroturfing rings,
and adversarial infiltrators — using only the signals it actively uncovers
via investigation actions.

---

## Quickstart

### Docker (recommended)

```bash
# Build
docker build -t social-graph-env .

# Run task_01 (default)
docker run -p 8000:8000 social-graph-env

# Run a specific task and seed
docker run -p 8000:8000 -e TASK_ID=task_02 -e SEED=123 social-graph-env
```

Verify the server is healthy:

```bash
curl http://localhost:8000/health
```

### Local (Python 3.11+)

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Connecting with the OpenEnv HTTP Client

```python
from openenv.core.client import HTTPEnvClient
from models import InvestigationAction, GraphObservation, ActionType

client = HTTPEnvClient(
    base_url="http://localhost:8000",
    action_class=InvestigationAction,
    observation_class=GraphObservation,
)

# Start an episode
obs = client.reset()
print(f"Visible nodes: {len(obs.nodes)}, step budget: {obs.step_budget}")

# Query a neighbour
action = InvestigationAction(
    action_type=ActionType.QUERY_NEIGHBORHOOD,
    target_ids=[obs.nodes[0].id],
    reasoning="Exploring the initial seed node.",
)
obs, reward, done, info = client.step(action)
print(f"reward={reward:.4f}  done={done}  f1={info.get('f1', 0):.3f}")

# Flag a suspected bot
flag_action = InvestigationAction(
    action_type=ActionType.FLAG_ACCOUNT,
    target_ids=["ACC_BOT_000", "ACC_BOT_001"],
    confidence=0.95,
    reasoning="Dense mutual follow-clique detected.",
)
obs, reward, done, info = client.step(flag_action)

# Submit the final report
submit = InvestigationAction(action_type=ActionType.SUBMIT_REPORT)
obs, reward, done, info = client.step(submit)
print(f"Final reward={reward:.4f}  F1={info['f1']:.3f}")
```

---

## API Endpoints

| Method | Path         | Description                                      |
|--------|--------------|--------------------------------------------------|
| GET    | `/health`    | Liveness probe (returns `{"status": "ok"}`)      |
| POST   | `/reset`     | Start a new episode; returns `GraphObservation`  |
| POST   | `/step`      | Submit an `InvestigationAction`; returns `(obs, reward, done, info)` |
| GET    | `/state`     | Read current observation without state mutation  |

---

## Tasks

### Task 01 · Bot Farm Detection · Easy

- **Graph**: 500 accounts (450 organic + 50 bots)
- **CIB pattern**: Dense follow-clique among bots; bots follow organic accounts
- **Step budget**: 20
- **Signal**: High in-degree clustering within bot cluster

### Task 02 · Astroturfing Ring Identification · Medium

- **Graph**: 2 000 accounts (1 910 organic + 3 × 30 rings)
- **CIB pattern**: Coordinated retweet bursts across three rings; organic-looking follows
- **Step budget**: 40
- **Signal**: Weight-2.5 retweet edges, intra-ring density

### Task 03 · Adversarial CIB with Infiltration · Hard

- **Graph**: 1 000 accounts (900 organic + 100 infiltrators)
- **CIB pattern**: Infiltrators mimic organic follows but coordinate via narrow temporal mention bursts
- **Step budget**: 80
- **Signal**: Only detectable via `REQUEST_TIMESERIES` + temporal window analysis

---

## Action Space

| Action                | Description                                             | Reward signal                        |
|-----------------------|---------------------------------------------------------|--------------------------------------|
| `QUERY_NEIGHBORHOOD`  | Expands visible subgraph around target account(s)       | +0.05 new node; −0.02 revisit; −0.30 3rd+ visit |
| `FLAG_ACCOUNT`        | Marks account(s) as inauthentic                         | +0.10 correct; −0.15 false positive  |
| `REQUEST_TIMESERIES`  | Requests temporal activity breakdown                    | +0.02 flat                           |
| `SUBMIT_REPORT`       | Terminates episode; triggers F1 scoring                 | See table below                      |

### Terminal F1 bonus (SUBMIT_REPORT)

| F1 score   | Bonus  |
|------------|--------|
| ≥ 0.90     | +0.50  |
| ≥ 0.75     | +0.30  |
| ≥ 0.50     | +0.15  |
| < 0.50     |  0.00  |
| + efficiency (steps < max) | +0.10 |

---

## Observation Space (`GraphObservation`)

```
nodes            List[AccountNode]  – accounts in the visible subgraph
edges            List[EdgeRecord]   – interactions in the visible subgraph
posts            List[PostRecord]   – posts from visible accounts (≈ 60 %)
temporal_windows List[TimeWindow]   – 4 × 6-hour activity windows
graph_stats      GraphStats         – global graph metrics (always visible)
step_budget      int                – remaining steps
reward           float              – reward from last action
done             bool               – episode completion flag
info             dict               – step count, F1 metrics
```

---

## Baseline Performance

| Task    | F1 (LLM baseline) |
|---------|-------------------|
| task_01 | ~0.74             |
| task_02 | ~0.51             |
| task_03 | ~0.29             |

Task 03's sharp drop demonstrates the need for temporal-graph reasoning agents
beyond vanilla LLMs.

---

## Running the Baseline Inference Script

```bash
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Optional: point at the HF Inference Providers API
export API_BASE_URL="https://api-inference.huggingface.co/v1"

python inference.py
```

---

## Validation

```bash
chmod +x validate-submission.sh
./validate-submission.sh http://localhost:8000
```

The script:
1. Hits `/health` to verify the server is up.
2. Calls `/reset` and checks the response schema.
3. Fires a `/step` with a `QUERY_NEIGHBORHOOD` action.
4. Fires a `/step` with a `SUBMIT_REPORT` action.
5. Prints PASS / FAIL for each check.
