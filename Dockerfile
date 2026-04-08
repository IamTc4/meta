FROM python:3.11-slim

WORKDIR /app

# Copy requirement files first
COPY requirements.txt /app/

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . /app/

# We'll default to running inference.py when container starts. 
# Depending on how the Meta PyTorch OpenEnv actually spins up HF spaces, 
# it might run a specific command or server. Let's assume standard python entry point
# or potentially an openenv CLI if we had it.
CMD ["python", "inference.py"]
