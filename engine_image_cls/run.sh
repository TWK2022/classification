#! /bin/bash
# 启动FlaskAPP
gunicorn -c gunicorn.conf -b 0.0.0.0:${SERVICE_PORT} run:app
