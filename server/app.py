"""
server/app.py – FastAPI application entry point.

Wires the SocialGraphEnv with openenv-core's create_fastapi_app factory.
Supports selecting the task via the TASK_ID environment variable
(defaults to 'task_01').

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so top-level modules are importable.
# This handles both `uvicorn server.app:app` (run from project root) and
# running the file directly in a sub-shell.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openenv.core.env_server import create_fastapi_app  # type: ignore

from models import GraphObservation, InvestigationAction
from server.environment import SocialGraphEnv

# ---------------------------------------------------------------------------
# Environment configuration from environment variables
# ---------------------------------------------------------------------------
TASK_ID: str = os.getenv("TASK_ID", "task_01")
SEED: int = int(os.getenv("SEED", "42"))

# Instantiate the environment (create_fastapi_app also accepts a class, but
# passing a pre-built instance lets us inject task_id / seed from env vars).
_env = SocialGraphEnv(task_id=TASK_ID, seed=SEED)

# ---------------------------------------------------------------------------
# Create the FastAPI application
# ---------------------------------------------------------------------------
app = create_fastapi_app(_env, InvestigationAction, GraphObservation)
