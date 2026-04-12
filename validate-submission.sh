#!/usr/bin/env bash
# =============================================================================
# validate-submission.sh — OpenEnv Submission Validator
# =============================================================================
#
# Builds the Docker image locally and validates the three required endpoints:
#   GET  /health
#   POST /reset
#   POST /step  (QUERY_NEIGHBORHOOD)
#   POST /step  (SUBMIT_REPORT)
#
# Usage:
#   chmod +x validate-submission.sh
#   ./validate-submission.sh [base_url] [repo_dir]
#
#   base_url  : URL of a running server (default: http://localhost:8000)
#               If 'docker', the script builds + runs the image itself.
#   repo_dir  : Path to the project root   (default: current directory)
#
# Examples:
#   ./validate-submission.sh                          # test local server on :8000
#   ./validate-submission.sh docker                  # build docker + test
#   ./validate-submission.sh https://my.hf.space     # test remote HF Space
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  BOLD='\033[1m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

FAILURES=0

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
BASE_URL="${1:-http://localhost:8000}"
REPO_DIR="${2:-.}"
IMAGE_TAG="social-graph-env-validate"
CONTAINER_NAME="sgenv-validate-$$"

# ---------------------------------------------------------------------------
# Optional Docker build + run
# ---------------------------------------------------------------------------
if [ "$BASE_URL" = "docker" ]; then
  info "Building Docker image '${IMAGE_TAG}' from '${REPO_DIR}' …"
  docker build -t "$IMAGE_TAG" "$REPO_DIR" || { fail "Docker build failed"; exit 1; }

  info "Starting container '${CONTAINER_NAME}' …"
  docker run -d --name "$CONTAINER_NAME" -p 8000:8000 "$IMAGE_TAG"

  # Wait for the server to become healthy (up to 30 s)
  HEALTHY=0
  for i in $(seq 1 15); do
    sleep 2
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "none")
    if [ "$STATUS" = "healthy" ]; then
      HEALTHY=1
      break
    fi
    info "Waiting for container to be healthy (attempt $i/15) …"
  done

  if [ $HEALTHY -eq 0 ]; then
    fail "Container did not reach 'healthy' state within 30 s"
    docker logs "$CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
    exit 1
  fi

  BASE_URL="http://localhost:8000"
  trap 'info "Stopping container …"; docker rm -f '"$CONTAINER_NAME"' >/dev/null 2>&1' EXIT
fi

info "Validating server at ${BASE_URL} …"
echo ""

# ---------------------------------------------------------------------------
# Helper: HTTP request with output
# ---------------------------------------------------------------------------
http_get() {
  curl -s -o /tmp/sgenv_resp.json -w "%{http_code}" "$1"
}
http_post() {
  curl -s -o /tmp/sgenv_resp.json -w "%{http_code}" \
    -X POST -H "Content-Type: application/json" -d "$2" "$1"
}

# ---------------------------------------------------------------------------
# 1. GET /health
# ---------------------------------------------------------------------------
echo -e "${BOLD}── 1. GET /health ──────────────────────────────────────${NC}"
CODE=$(http_get "${BASE_URL}/health")
if [ "$CODE" = "200" ]; then
  pass "/health returned 200"
  cat /tmp/sgenv_resp.json; echo ""
else
  fail "/health returned HTTP $CODE"
  cat /tmp/sgenv_resp.json; echo ""
fi
echo ""

# ---------------------------------------------------------------------------
# 2. POST /reset
# ---------------------------------------------------------------------------
echo -e "${BOLD}── 2. POST /reset ──────────────────────────────────────${NC}"
CODE=$(http_post "${BASE_URL}/reset" '{}')
if [ "$CODE" = "200" ]; then
  pass "/reset returned 200"
  # Check required fields exist in response
  for field in nodes edges graph_stats step_budget done; do
    if grep -q "\"${field}\"" /tmp/sgenv_resp.json; then
      pass "  Field '${field}' present in GraphObservation"
    else
      fail "  Field '${field}' MISSING from GraphObservation"
    fi
  done
else
  fail "/reset returned HTTP $CODE"
  cat /tmp/sgenv_resp.json; echo ""
fi
echo ""

# ---------------------------------------------------------------------------
# 3. POST /step  (QUERY_NEIGHBORHOOD)
# ---------------------------------------------------------------------------
echo -e "${BOLD}── 3. POST /step (QUERY_NEIGHBORHOOD) ─────────────────${NC}"
STEP_PAYLOAD='{"action_type":"QUERY_NEIGHBORHOOD","target_ids":[],"confidence":1.0,"reasoning":"validate"}'
CODE=$(http_post "${BASE_URL}/step" "$STEP_PAYLOAD")
if [ "$CODE" = "200" ]; then
  pass "/step QUERY_NEIGHBORHOOD returned 200"
  # OpenEnv step returns [obs, reward, done, info]
  if grep -q "step_budget" /tmp/sgenv_resp.json; then
    pass "  Response contains step_budget (valid GraphObservation)"
  else
    fail "  Response does not appear to be a valid 4-tuple / GraphObservation"
    cat /tmp/sgenv_resp.json; echo ""
  fi
else
  fail "/step returned HTTP $CODE"
  cat /tmp/sgenv_resp.json; echo ""
fi
echo ""

# ---------------------------------------------------------------------------
# 4. POST /step  (SUBMIT_REPORT)
# ---------------------------------------------------------------------------
echo -e "${BOLD}── 4. POST /step (SUBMIT_REPORT) ───────────────────────${NC}"
SUBMIT_PAYLOAD='{"action_type":"SUBMIT_REPORT","target_ids":[],"confidence":1.0,"reasoning":"validate"}'
CODE=$(http_post "${BASE_URL}/step" "$SUBMIT_PAYLOAD")
if [ "$CODE" = "200" ]; then
  pass "/step SUBMIT_REPORT returned 200"
else
  fail "/step SUBMIT_REPORT returned HTTP $CODE"
  cat /tmp/sgenv_resp.json; echo ""
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo -e "${BOLD}── Summary ──────────────────────────────────────────────${NC}"
if [ $FAILURES -eq 0 ]; then
  echo -e "${GREEN}${BOLD}All checks passed! ✅${NC}"
  exit 0
else
  echo -e "${RED}${BOLD}${FAILURES} check(s) failed. ❌${NC}"
  exit 1
fi
