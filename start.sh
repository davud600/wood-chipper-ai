#!/bin/bash

if [ "$1" == "install" ]; then
  echo "Running apt update and install..."
  apt update && apt install redis-server -y
fi

. .venv/bin/activate

redis-server --daemonize yes

: > gunicorn_output.log

(
  gunicorn -w 1 -b 0.0.0.0:8001 server:app \
    --timeout 600 \
    --threads 1 \
    --access-logfile gunicorn_output.log \
    --error-logfile gunicorn_output.log \
    --capture-output \
    --log-level debug \
    # --reload # dev
) &

echo "server started in the background with PID $!"
