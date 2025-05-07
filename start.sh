#!/bin/bash

source . .venv/bin/activate
redis-server --daemonize yes
gunicorn -w 1 -b 0.0.0.0:8000 server:app --timeout 600 --threads 1
