#!/bin/bash
exec >> /Users/ojacie/.semantic-index/logs/watcher-stderr.log 2>&1
echo "=== Starting watcher at $(date) ==="
export PATH="/Users/ojacie/.semantic-index/.venv/bin:/usr/local/bin:/usr/bin:/bin"
export VIRTUAL_ENV="/Users/ojacie/.semantic-index/.venv"
export PYTHONPATH="/Users/ojacie/.semantic-index"
export HOME="/Users/ojacie"
cd /Users/ojacie/.semantic-index
exec /Users/ojacie/.semantic-index/.venv/bin/python3 /Users/ojacie/.semantic-index/watcher.py
