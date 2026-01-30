#!/bin/bash
# CAN bus activation script for Piper robotic arm
# Usage: bash can_activate.sh [can_interface] [baudrate]
#
# Example:
#   bash can_activate.sh can0 1000000

set -euo pipefail

CAN_IF="${1:-can0}"
BAUDRATE="${2:-1000000}"

echo "[INFO] Activating CAN interface: $CAN_IF with baudrate: $BAUDRATE"

# Bring down the interface first (ignore errors if already down)
sudo ip link set "$CAN_IF" down 2>/dev/null || true

# Set CAN type and bitrate
sudo ip link set "$CAN_IF" type can bitrate "$BAUDRATE"

# Bring up the interface
sudo ip link set "$CAN_IF" up

# Verify the interface is up
if ip link show "$CAN_IF" | grep -q "UP"; then
    echo "[OK] CAN interface $CAN_IF is now active"
    ip -details link show "$CAN_IF" | grep -E "(state|bitrate)"
else
    echo "[ERROR] Failed to activate CAN interface $CAN_IF" >&2
    exit 1
fi
