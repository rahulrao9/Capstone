#!/bin/bash

# Redirect all output to a log file
exec >> /app/daily_tasks.log 2>&1

echo "Starting daily tasks at $(date)"

echo "Running main.py..."
/usr/local/bin/python /app/main.py

echo "Running downlatestimg.sh..."
/app/downlatestimg.sh

echo "Running driveUploader.py..."
/usr/local/bin/python /app/driveUploader.py

echo "All tasks completed at $(date)"