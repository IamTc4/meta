import asyncio
import os
import textwrap
from typing import List, Optional
import random

from openai import OpenAI

from env import SocialGraphEnv
from models import InvestigationAction, ActionType

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "social-graph-env")

def mock_llm_chain(client, obs, model=MODEL_NAME):
    """
    In a real scenario, this communicates with OpenAI's API format that routes
    to HF spaces or bedrock. Here we provide a heuristic-mock response just
    for checking functionality.
    """
    if obs.step_budget <= 1:
        return InvestigationAction(action_type=ActionType.SUBMIT_REPORT, reasoning="Budget ending.")
        
    if random.random() > 0.5 and obs.nodes:
        n = random.choice(obs.nodes)
        return InvestigationAction(action_type=ActionType.FLAG_ACCOUNT, target_ids=[n.id], confidence=0.8)
    
    if obs.nodes:
        n = random.choice(obs.nodes)
        return InvestigationAction(action_type=ActionType.QUERY_NEIGHBORHOOD, target_ids=[n.id])
        
    return InvestigationAction(action_type=ActionType.SUBMIT_REPORT)

def run_inference():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or "dummy_key"
    )

    tasks = ["task_01", "task_02", "task_03"]
    
    for task in tasks:
        env = SocialGraphEnv(task_id=task)
        obs = env.reset()
        done = False
        
        rewards_list = []
        steps = 0
        error_msg = "null"
        success = False
        score = 0.0
        
        print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")
        
        try:
            while not done:
                action = mock_llm_chain(client, obs)
                
                try:
                    obs, reward, done, info = env.step(action)
                    error_msg = "null"
                except Exception as e:
                    reward = 0.0
                    done = True
                    error_msg = str(e).replace("\n", " ")
                    info = {"step": steps + 1}
                
                steps = info.get("step", steps + 1)
                rewards_list.append(reward)
                
                action_str = action.model_dump_json().replace("\n", " ")
                
                done_str = "true" if done else "false"
                print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}")
                
            total_reward = sum(rewards_list)
            score = max(0.0, min(1.0, total_reward))
            
            success = score > 0.0

        except Exception as e:
            pass
            
        finally:
            success_str = "true" if success else "false"
            rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
            if not rewards_str: 
                rewards_str = "0.00"
            print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    run_inference()
