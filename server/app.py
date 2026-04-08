from openenv.core.env_server.http_server import create_app
import uvicorn
import argparse
import os

try:
    from models import InvestigationAction, GraphObservation
    from env import SocialGraphEnv
except ImportError:
    from ..models import InvestigationAction, GraphObservation
    from ..env import SocialGraphEnv

# Create the app with web interface and README integration
app = create_app(
    SocialGraphEnv,
    InvestigationAction,
    GraphObservation,
    env_name="social_graph_env",
    max_concurrent_envs=1,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
