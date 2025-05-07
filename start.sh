#!/bin/bash

source .venv/bin/activate
redis-server --daemonize yes
gunicorn -w 1 -b 0.0.0.0:8001 server:app --timeout 600 --threads 1 --daemon --access-logfile output.log --error-logfile output.log
