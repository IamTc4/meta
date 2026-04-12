"""
env.py – Backward-compatibility shim.

The canonical implementation now lives in server/environment.py.
All imports from this module work as before, but the class now inherits
from openenv.core.env_server.Environment and is fully OpenEnv-compliant.
"""

from server.environment import SocialGraphEnv  # noqa: F401 (re-export)

__all__ = ["SocialGraphEnv"]
