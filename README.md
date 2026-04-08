---
title: Social Graph Env
emoji: 🔊
colorFrom: indigo
colorTo: indigo
sdk: docker
app_port: 8000
---

# Social Graph Manipulation Detection Environment

## Overview
This is an OpenEnv-compliant environment designed for the **Meta PyTorch OpenEnv Hackathon**. The environment tasks models with identifying and correctly flagging Coordinated Inauthentic Behavior (CIB) including bot farms, astroturfing rings, and adversarial infiltration strategies in simulated social networks.

Instead of reading text, agents must parse through typed graph snapshots (nodes, edges, timestamps) utilizing Pydantic. It's built for realism, determinism, and precision testing. 

## Action & Observation Spaces
### Observations
A typed `GraphObservation` is emitted every step. It includes collections of `AccountNode` and `EdgeRecord` to construct subgraph data, while hiding ground truth labels. 
Metrics including global `GraphStats` and `TimeWindow`s allow agents to measure temporal anomaly rates.

### Actions
Agents output `InvestigationAction`, consisting of:
- `FLAG_ACCOUNT`: Labels a target as inauthentic (precision is essential).
- `QUERY_NEIGHBORHOOD`: Retrieves the interactions expanding a target.
- `REQUEST_TIMESERIES`: Requests temporal interaction spikes.
- `SUBMIT_REPORT`: Finishes the episode.

## Tasks
1. **Task 01 (Easy): Bot Farm Detection** - Extract a 50-account cluster from a 500-account graph. Clear temporal and logic signatures.
2. **Task 02 (Medium): Astroturfing Ring** - Identify 3-4 coordinated clusters in a 2000-account network with organic-looking interweaved engagement.
3. **Task 03 (Hard): Adversarial CIB** - Highly difficult infiltration with history laundering in a 5000-account network. 

## Hugging Face Token Guide
The baseline inference script connects to language models and OpenEnv systems via the Hugging Face API format. To use the environment effectively, you will need a valid `HF_TOKEN`.

### Step 1: Generate an HF Token
1. Log in or sign up at [Hugging Face](https://huggingface.co).
2. Navigate to your settings: **Profile Icon -> Settings -> Access Tokens**.
3. Choose **New token**. You can use a classic `Read` token, or a **Fine-grained** token (recommended for better security). If using a fine-grained token, ensure it has permissions to access inference endpoints or read public models, depending on your setup.
4. Copy the token starting with `hf_...` and keep it secure.

### Step 2: Configure the Token
You must export the token as an environment variable before running the setup.

**On Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_your_token_here"
```

**On Linux/macOS:**
```bash
export HF_TOKEN="hf_your_token_here"
```

## Installation & Usage

### Docker (Recommended)
This environment is designed as a portable HF Space. Make sure your environment variable is set as shown above.
```bash
cd social-graph-env
docker build -t sgmd .
docker run -e HF_TOKEN=$HF_TOKEN sgmd
```

### Local Testing
Ensure you have Python 3.11+.
```bash
pip install -r requirements.txt
python inference.py
```

## Baseline Performance
Using generic prompt instructions (see PRD for details):
- Task 01: F1 Range ~0.74 
- Task 02: F1 Range ~0.51
- Task 03: F1 Range ~0.29

The drastic drop in Task 3 showcases the need for sophisticated temporal graph reasoning agents beyond baseline LLMs.
