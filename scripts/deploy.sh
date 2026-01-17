#!/bin/bash
# Deploy script to sync project to Orin target
# Usage: bash deploy.sh [user@host] [remote_path]
#
# Example:
#   bash deploy.sh nvidia@orin.local
#   bash deploy.sh nvidia@192.168.1.100 /home/nvidia/projects/arm_demo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

TARGET="${1:-nvidia@orin.local}"
REMOTE_PATH="${2:-~/arm_demo}"

echo "[INFO] Deploying from: $PROJECT_DIR"
echo "[INFO] Target: $TARGET:$REMOTE_PATH"

# Rsync with common options:
#   -a: archive mode (preserves permissions, timestamps, etc.)
#   -v: verbose
#   -z: compress during transfer
#   --delete: remove files on target that don't exist locally
#   --exclude: skip unwanted files
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '*.egg-info' \
    "$PROJECT_DIR/" "$TARGET:$REMOTE_PATH/"

echo "[OK] Deployment complete"
echo ""
echo "Next steps on target ($TARGET):"
echo "  1. cd $REMOTE_PATH"
echo "  2. bash scripts/can_activate.sh can0 1000000"
echo "  3. python examples/01_connect.py"
