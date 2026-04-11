import os
import random
import time
from typing import List, Optional
from openai import OpenAI
from env import SocialGraphEnv
from models import InvestigationAction, ActionType

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.5-preview")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# Timeout in seconds for each LLM call
LLM_TIMEOUT = 30 

def get_llm_action(client, obs, model=MODEL_NAME):
    """
    Calls the LLM to decide the next action based on current observations.
    Includes a timeout and robust fallback.
    """
    system_prompt = (
        "You are a Trust & Safety analyst. Your task is to detect Coordinated Inauthentic Behavior (CIB) "
        "in a social graph. You can query node neighborhoods, flag accounts, or submit a final report. "
        "Respond ONLY with an InvestigationAction in JSON format."
    )
    
    obs_json = obs.model_dump_json()
    prompt = f"Observations: {obs_json}\nDecide the next action."
    
    try:
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
        # Fallback to smart random action if model fails or times out
        if obs.step_budget <= 1:
            return InvestigationAction(action_type=ActionType.SUBMIT_REPORT, reasoning="Budget ending or timeout.")
        
        if obs.nodes:
            n = random.choice(obs.nodes)
            return InvestigationAction(
                action_type=ActionType.QUERY_NEIGHBORHOOD, 
                target_ids=[n.id],
                reasoning="Fallback action due to error/timeout."
            )
        return InvestigationAction(action_type=ActionType.SUBMIT_REPORT, reasoning="Fallback to submit.")

def run_inference_loop():
    # Benchmark name from environment variable or default
    benchmark = os.getenv("MY_ENV_V4_BENCHMARK", "social-graph-env")
    tasks = ["task_01", "task_02", "task_03"]
    
    for task_id in tasks:
        # One [START] line at episode begin
        print(f"[START] task={task_id} env={benchmark} model={MODEL_NAME}", flush=True)
        
        env = SocialGraphEnv(task_id=task_id)
        obs = env.reset()
        done = False
        
        rewards_list = []
        total_steps = 0
        success = False
        
        task_start_time = time.time()
        # 10 minute limit per individual task
        TASK_TIME_LIMIT = 600 
        
        try:
            while not done:
                total_steps += 1
                
                # Check if we are exceeding the per-task time limit
                if time.time() - task_start_time > TASK_TIME_LIMIT:
                    action = InvestigationAction(action_type=ActionType.SUBMIT_REPORT, reasoning="Time limit reached.")
                else:
                    action = get_llm_action(client, obs)
                
                error_msg = "null"
                try:
                    obs = env.step(action)
                    reward = obs.reward
                    done = obs.done
                except Exception as e:
                    reward = 0.0
                    done = True
                    error_msg = str(e).replace("\n", " ")
                
                rewards_list.append(reward)
                
                # One [STEP] line per step
                action_str = action.model_dump_json().replace("\n", " ")
                done_str = "true" if done else "false"
                print(f"[STEP] step={total_steps} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}", flush=True)

            # Evaluate success (example logic: some reward > 0)
            success = sum(rewards_list) > 0

        except Exception as e:
            # Errors handled in the loop, but catch unexpected ones
            pass
        finally:
            # One [END] line after episode (even on exception)
            success_str = "true" if success else "false"
            rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
            if not rewards_str: 
                rewards_str = "0.00"
            
            # CRITICAL: No extra fields like 'score=' here
            print(f"[END] success={success_str} steps={total_steps} rewards={rewards_str}", flush=True)
            
            # Attempt to close env if method exists
            if hasattr(env, 'close'):
                env.close()

if __name__ == "__main__":
    try:
        run_inference_loop()
    except Exception as e:
        # Final safety print to satisfy parser structure if possible
        # but the loop handles its own [END] lines
        pass

