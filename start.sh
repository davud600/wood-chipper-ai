#!/bin/bash

if [ "$1" == "install" ]; then
  echo "Running apt-get update and install..."
  apt-get update && apt-get install -y redis-server
fi

. .venv/bin/activate

redis-server --daemonize yes

: > server_output.log

(
  gunicorn -w 1 -b 0.0.0.0:8001 server:app \
    --timeout 600 \
    --threads 1 \
    --access-logfile server_output.log \
    --error-logfile server_output.log \
    --capture-output \
    --log-level debug \
    # --reload # dev
) &

echo "server started in the background with PID $!"
