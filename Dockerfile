FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . /app/

# Expose the FastAPI port
EXPOSE 8000

# Set environment variable to ensure logs are shown
ENV PYTHONUNBUFFERED=1

# Run the OpenEnv server
# We use -m server.app because it's a package
CMD ["python", "-m", "server.app"]
