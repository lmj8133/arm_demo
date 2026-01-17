#!/bin/bash
# CAN bus deactivation script for Piper robotic arm
# Usage: bash can_deactivate.sh [can_interface]
#
# Example:
#   bash can_deactivate.sh can0

set -euo pipefail

CAN_IF="${1:-can0}"

echo "[INFO] Deactivating CAN interface: $CAN_IF"

# Bring down the interface
sudo ip link set "$CAN_IF" down

# Verify the interface is down
if ip link show "$CAN_IF" | grep -q "DOWN"; then
    echo "[OK] CAN interface $CAN_IF is now deactivated"
else
    echo "[WARN] CAN interface $CAN_IF may still be active"
fi
