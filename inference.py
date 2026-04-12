"""
inference.py – Baseline LLM inference loop for Social Graph Manipulation Detection.

Connects to a language model via an OpenAI-compatible API and runs each of
the three environment tasks.  Output follows the OpenEnv evaluation format:

    [START] task=<id> env=<name> model=<model>
    [STEP]  step=<n> action=<json> reward=<r> done=<bool> error=<null|msg>
    [END]   success=<bool> steps=<n> rewards=<comma-separated>

Environment variables:
    HF_TOKEN       – (required) API key / HF token
    API_BASE_URL   – API base URL (default: https://api-inference.huggingface.co/v1)
    MODEL_NAME     – model identifier (default: meta-llama/Llama-3.1-8B-Instruct)
"""

import os
import random
import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from env import SocialGraphEnv
from models import ActionType, GraphObservation, InvestigationAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv(
    "API_BASE_URL", "https://api-inference.huggingface.co/v1"
)
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Timeout in seconds for each LLM call
LLM_TIMEOUT: int = 30

# Per-task wall-clock time limit (10 minutes)
TASK_TIME_LIMIT: int = 600


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def get_llm_action(obs: GraphObservation, model: str = MODEL_NAME) -> InvestigationAction:
    """
    Ask the LLM to choose the next investigation action.

    Falls back to a deterministic heuristic on any API error or timeout.
    """
    system_prompt = (
        "You are a Trust & Safety analyst. Your task is to detect Coordinated "
        "Inauthentic Behavior (CIB) in a social graph. You can:\n"
        "  - QUERY_NEIGHBORHOOD: expand the visible subgraph around an account.\n"
        "  - FLAG_ACCOUNT: mark accounts as inauthentic. Precision matters.\n"
        "  - REQUEST_TIMESERIES: request temporal activity details.\n"
        "  - SUBMIT_REPORT: finalise the episode.\n"
        "Respond ONLY with a valid JSON object matching the InvestigationAction schema."
    )
    prompt = f"Current observation:\n{obs.model_dump_json(indent=2)}\n\nChoose your next action."

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=LLM_TIMEOUT,
        )
        content = resp.choices[0].message.content
        return InvestigationAction.model_validate_json(content)
    except Exception:
        return _fallback_action(obs)


def _fallback_action(obs: GraphObservation) -> InvestigationAction:
    """Deterministic fallback when the LLM call fails."""
    if obs.step_budget <= 1:
        return InvestigationAction(
            action_type=ActionType.SUBMIT_REPORT,
            reasoning="Budget nearly exhausted – submitting report.",
        )
    if obs.nodes:
        target = random.choice(obs.nodes)
        return InvestigationAction(
            action_type=ActionType.QUERY_NEIGHBORHOOD,
            target_ids=[target.id],
            reasoning="Fallback: random neighbourhood expansion.",
        )
    return InvestigationAction(
        action_type=ActionType.SUBMIT_REPORT,
        reasoning="No nodes visible – submitting report.",
    )


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_task(task_id: str, benchmark: str) -> None:
    print(f"[START] task={task_id} env={benchmark} model={MODEL_NAME}", flush=True)

    env = SocialGraphEnv(task_id=task_id)
    obs = env.reset()
    done = False

    rewards_list: List[float] = []
    total_steps = 0
    success = False
    task_start = time.time()

    try:
        while not done:
            total_steps += 1

            # Wall-clock guard: force submit if approaching time limit
            if time.time() - task_start > TASK_TIME_LIMIT:
                action = InvestigationAction(
                    action_type=ActionType.SUBMIT_REPORT,
                    reasoning="Task time limit reached.",
                )
            else:
                action = get_llm_action(obs)

            error_msg = "null"
            try:
                # step() now returns (obs, reward, done, info)
                obs, reward, done, info = env.step(action)
            except Exception as exc:
                reward = 0.0
                done = True
                error_msg = str(exc).replace("\n", " ")

            rewards_list.append(reward)
            action_json = action.model_dump_json().replace("\n", " ")
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={total_steps} action={action_json} "
                f"reward={reward:.4f} done={done_str} error={error_msg}",
                flush=True,
            )

        success = sum(rewards_list) > 0

    except Exception:
        pass

    finally:
        rewards_str = ",".join(f"{r:.4f}" for r in rewards_list) or "0.0000"
        success_str = "true" if success else "false"
        print(
            f"[END] success={success_str} steps={total_steps} rewards={rewards_str}",
            flush=True,
        )
        if hasattr(env, "close"):
            env.close()


def run_inference_loop() -> None:
    benchmark = os.getenv("MY_ENV_V4_BENCHMARK", "social-graph-env")
    for task_id in ["task_01", "task_02", "task_03"]:
        run_task(task_id, benchmark)


if __name__ == "__main__":
    run_inference_loop()
