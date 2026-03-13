#!/bin/bash
exec >> /Users/ojacie/.semantic-index/logs/hayhooks-stderr.log 2>&1
echo "=== Starting hayhooks at $(date) ==="
export PATH="/Users/ojacie/.semantic-index/.venv/bin:/usr/local/bin:/usr/bin:/bin"
export VIRTUAL_ENV="/Users/ojacie/.semantic-index/.venv"
export PYTHONPATH="/Users/ojacie/.semantic-index"
export HOME="/Users/ojacie"
cd /Users/ojacie/.semantic-index
exec /Users/ojacie/.semantic-index/.venv/bin/hayhooks mcp run \
  --pipelines-dir /Users/ojacie/.semantic-index/pipelines \
  --additional-python-path /Users/ojacie/.semantic-index
