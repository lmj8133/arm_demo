# Project CLAUDE.md

> **Project Context**
> Orin-based vision + robotic arm control system. Develop locally (WSL), deploy via `rsync`, test on target.

---

## 0) Project Setup

* **Target platform**: NVIDIA Orin (ARM64, JetPack)
* **Hardware**: USB camera (vision) + CAN bus (robotic arm control)
* **Workflow**: Local dev on WSL → `rsync` to Orin → test on target
* Code must be **cross-platform** (x64 dev env, ARM64 target)

---

## 1) Piper Robotic Arm Resources

### SDK (Python)

* **Repo**: <https://github.com/agilexrobotics/piper_sdk>
* **Install**: `pip3 install piper_sdk` (requires `python-can>=3.3.4`)
* **Supported OS**: Ubuntu 18.04/20.04/22.04 with Python 3.6/3.8/3.10

**Quick Start**:

```python
from piper_sdk import *

piper = C_PiperInterface(can_name="can0", dh_is_offset=1)
piper.ConnectPort()

# Switch to slave mode for feedback
piper.MasterSlaveConfig(0xFC, 0, 0, 0)

# Read joint angles
print(piper.GetArmJointMsgs())
```

**Key Features**:

* Real-time joint angle feedback
* Motor control & firmware detection
* Master-slave mode configuration
* Dual-arm support (independent CAN modules)
* SDK-level joint/gripper position limiting

---

### ROS2 (Foxy)

* **Repo**: <https://github.com/agilexrobotics/Piper_ros/tree/ros-foxy-no-aloha>
* **Branch**: `ros-foxy-no-aloha`

**Dependencies**:

```bash
pip3 install python-can scipy piper_sdk
sudo apt install ros-foxy-ros2-control ros-foxy-ros2-controllers ros-foxy-controller-manager
```

**Topics**:

| Topic                  | Type                         | Description             |
| ---------------------- | ---------------------------- | ----------------------- |
| `/arm_status`          | `PiperStatusMsg`             | Arm state feedback      |
| `/end_pose`            | —                            | End-effector pose       |
| `/joint_states_single` | `sensor_msgs/JointState`     | Joint feedback          |
| `/joint_states`        | `sensor_msgs/JointState`     | Joint command (sub)     |

**Services**:

| Service       | Type                    | Description          |
| ------------- | ----------------------- | -------------------- |
| `/enable_srv` | `piper_msgs/srv/Enable` | Enable/disable arm   |

**Launch**:

```bash
ros2 launch piper start_single_piper.launch.py can_port:=can0 auto_enable:=false
```

**Enable Arm**:

```bash
ros2 service call /enable_srv piper_msgs/srv/Enable "enable_request: true"
```

**Publish Joint Command**:

```bash
ros2 topic pub /joint_states sensor_msgs/msg/JointState \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, \
    name: ['joint1','joint2','joint3','joint4','joint5','joint6','joint7'], \
    position: [0.2, 0.2, -0.2, 0.3, -0.2, 0.5, 0.01]}"
```

**RViz Simulation**:

```bash
ros2 launch piper_description display_xacro.launch.py
```

---

## 2) CAN Bus Setup

**Single CAN module** (baud rate 1000000):

```bash
bash can_activate.sh can0 1000000
```

**Multiple CAN modules**: Configure USB port mapping in `can_config.sh` or `can_muti_activate.sh`.

**Important**:

* Always activate CAN and set baud rate **before** running SDK/ROS2 nodes
* First CAN message frames contain default values (typically 0)
* Robot must be in **slave mode** (`0xFC`) to read feedback
