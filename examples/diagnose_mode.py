"""Diagnose arm control mode status.

Use this script to check the current control mode after entering
teaching mode from the remote controller.

Usage:
    uv run python examples/diagnose_mode.py
"""
from piper_sdk import C_PiperInterface_V2
import time


def main():
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    time.sleep(0.2)

    status = piper.GetArmStatus()
    ctrl_mode = status.arm_status.ctrl_mode
    arm_status_val = status.arm_status.arm_status
    teach_status = status.arm_status.teach_status

    # Handle both enum and int types
    ctrl_mode_val = ctrl_mode.value if hasattr(ctrl_mode, 'value') else int(ctrl_mode)
    arm_status_int = arm_status_val.value if hasattr(arm_status_val, 'value') else int(arm_status_val)
    teach_status_val = teach_status.value if hasattr(teach_status, 'value') else int(teach_status)

    print("=== Arm Status ===")
    print(f"ctrl_mode:    0x{ctrl_mode_val:02X} ({ctrl_mode})")
    print(f"arm_status:   0x{arm_status_int:02X} ({arm_status_val})")
    print(f"teach_status: 0x{teach_status_val:02X} ({teach_status})")
    print()
    print("ctrl_mode reference:")
    print("  0x00 = STANDBY")
    print("  0x01 = CAN_CTRL (normal)")
    print("  0x02 = TEACHING_MODE")
    print("  0x05 = REMOTE_CONTROL_MODE")


if __name__ == "__main__":
    main()
