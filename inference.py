import asyncio
import os
import textwrap
from typing import List, Optional
import random
import time

from openai import OpenAI

from env import SocialGraphEnv
from models import InvestigationAction, ActionType

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "social-graph-env")

# Timeout in seconds for each LLM call
LLM_TIMEOUT = 30 

def get_llm_action(client, obs, model=MODEL_NAME):
    """
    Calls the LLM to decide the next action based on current observations.
    Includes a timeout and robust fallback.
    """
    system_prompt = "You are a Trust & Safety analyst. Your task is to detect Coordinated Inauthentic Behavior (CIB) in a social graph. You can query node neighborhoods, flag accounts, or submit a final report. Respond ONLY with an InvestigationAction in JSON format."
    
    # Prune observations if they are too large to avoid context window issues and slow processing
    # Keeping only a subset of nodes/edges if budget is tight or list is huge
    obs_json = obs.model_dump_json()
    if len(obs_json) > 10000:
        # Simple pruning: just keep latest nodes/edges if we were doing complex logic, 
        # but here we'll just log and rely on the model handling it or the timeout.
        pass

    prompt = f"Observations: {obs_json}\nDecide the next action."
    
    try:
        # Note: openai-python v1.x supports 'timeout' parameter in create()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            timeout=LLM_TIMEOUT
        )
        content = response.choices[0].message.content
        return InvestigationAction.model_validate_json(content)
    except Exception as e:
        print(f"LLM Error or Timeout: {e}")
        # Fallback to smart random action if model fails or times out
        if obs.step_budget <= 1:
            return InvestigationAction(action_type=ActionType.SUBMIT_REPORT, reasoning="Budget ending or timeout.")
        
        if obs.nodes:
            # Pick a node we haven't investigated deeply or just a random one
            n = random.choice(obs.nodes)
            return InvestigationAction(
                action_type=ActionType.QUERY_NEIGHBORHOOD, 
                target_ids=[n.id],
                reasoning="Fallback action due to error/timeout."
            )
        return InvestigationAction(action_type=ActionType.SUBMIT_REPORT, reasoning="Fallback to submit.")

async def run_inference():
    if not API_KEY:
        print("WARNING: HF_TOKEN not set. Inference will likely fail unless using a local mock. Using 'dummy_key'.")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or "dummy_key"
    )

    tasks = ["task_01", "task_02", "task_03"]
    
    for task in tasks:
        print(f"\n[START] task={task} env={BENCHMARK} model={MODEL_NAME}")
        env = SocialGraphEnv(task_id=task)
        obs = env.reset()
        done = False
        
        rewards_list = []
        steps = 0
        error_msg = "null"
        success = False
        score = 0.0
        
        task_start_time = time.time()
        # 10 minute limit per individual task to stay well within 30 min total
        TASK_TIME_LIMIT = 600 
        
        try:
            while not done:
                # Check if we are exceeding the per-task time limit
                if time.time() - task_start_time > TASK_TIME_LIMIT:
                    print(f"Task {task} exceeded time limit. Forcing submission.")
                    action = InvestigationAction(action_type=ActionType.SUBMIT_REPORT, reasoning="Time limit reached.")
                else:
                    # Run LLM call in a thread if we were using a non-async client, 
                    # but simple blocking with timeout is often sufficient if the loop isn't massive.
                    action = get_llm_action(client, obs)
                
                try:
                    # env.step is synchronous in the current implementation
                    obs = env.step(action)
                    reward = obs.reward
                    done = obs.done
                    info = obs.info
                    error_msg = "null"
                except Exception as e:
                    print(f"Env Step Error: {e}")
                    reward = 0.0
                    done = True
                    error_msg = str(e).replace("\n", " ")
                    info = {"step": steps + 1}
                
                steps = info.get("step", steps + 1)
                rewards_list.append(reward)
                
                action_str = action.model_dump_json().replace("\n", " ")
                done_str = "true" if done else "false"
                print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}")
                
                # Small sleep to yield control if needed, though not strictly necessary in this blocking loop
                await asyncio.sleep(0.01)

            total_reward = sum(rewards_list)
            score = max(0.0, min(1.0, total_reward))
            success = score > 0.0

        except Exception as e:
            print(f"Unexpected error in task loop: {e}")
            
        finally:
            success_str = "true" if success else "false"
            rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
            if not rewards_str: 
                rewards_str = "0.00"
            print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    asyncio.run(run_inference())
