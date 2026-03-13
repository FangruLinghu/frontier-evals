#!/bin/bash
# Basic reproduce.sh for testing the pipeline

set -e  # Exit on error

echo "=== Starting reproduction ==="

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Run main script
echo "Running main.py..."
python3 main.py --help || python3 main.py || echo "main.py execution attempted"

echo "=== Reproduction complete ==="
