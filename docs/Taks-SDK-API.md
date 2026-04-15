# Taks SDK API 文档

Taks SDK 是 Taks 人形机器人的 Python 控制库，提供三个核心子模块：

| 模块 | 功能 | 依赖真机 |
|------|------|----------|
| `taks.client` | 连接管理、电机控制、夹爪、IMU | 是 |
| `taks.ik` | 逆运动学求解（末端→关节角） | 否 |
| `taks.vr` | VR遥操作（UDP接收、偏移管理） | 否 |

## 参考Prompt

```
根据"/home/xhz/OpenFigure/taks/README.md"的API描述，结合"/home/xhz/OpenFigure/example/demo_ik_circle.py"，"/home/xhz/OpenFigure/example/demo_ik_keyboard.py"，"/home/xhz/OpenFigure/example/demo_ik_vr.py"的example，里利用taks库写一个左右手在YZ平面上画正方形，X轴保持不变的demo，写到/home/xhz/OpenFigure/example里
```

**注意："......"需要换成本机代码的绝对路径**

## 安装

### 基础依赖（py310测试完美）

```bash
pip install numpy eclipse-zenoh mujoco mujoco-python-viewer mink loop-rate-limiters orjson uvloop scipy rich
```

### 未来方法（暂不可用）

```bash
pip install taks
```

## 极简示例

```python
from taks.client import TaksClient
from taks.ik import IKController
from taks.vr import VRController
from loop_rate_limiters import RateLimiter

ik = IKController()
vr = VRController()
vr.start()

client = TaksClient("Semi-Taks-T1", host="192.168.5.12", with_gripper=True)
client.connect(wait_data=True)

rate = RateLimiter(frequency=50)
try:
    while True:
        targets = vr.step(ik)       # VR数据→末端目标
        cmd = ik.step(**targets)    # 末端目标→MIT命令
        client.controlMIT(cmd)      # 发送到真机
        rate.sleep()
finally:
    vr.stop()
    client.close()
```

## 支持的设备型号

| 型号 | 说明 | 关节数 |
|------|------|--------|
| `Taks-T1` | 全身机器人 | 34 |
| `Semi-Taks-T1` | 半身机器人（双臂+腰+颈） | 22 |

---

## 安装与路径

SDK 位于 `backend/libs/SDK/taks/` 目录下。使用前需将 SDK 路径加入 `sys.path`：

```python
import sys, os
_curr = os.path.abspath(__file__)
while _curr != '/' and os.path.basename(_curr) != 'backend':
    _curr = os.path.dirname(_curr)
sys.path.insert(0, _curr)
sys.path.insert(0, os.path.join(_curr, 'libs', 'SDK'))
```

---

## 一、taks.client — 连接与电机控制

### 1.1 快速开始

```python
from taks.client import TaksClient

# 推荐：使用 TaksClient（asynccontextmanager 管理生命周期）
client = TaksClient("Semi-Taks-T1", host="192.168.5.12", with_gripper=True)
client.connect(wait_data=True, timeout=5.0)
client.controlMIT({1: {'q': 0.5, 'kp': 10, 'kd': 1}})
state = client.get_state()
client.close()

# 或使用 with 语句
with TaksClient("Semi-Taks-T1", host="192.168.5.12") as client:
    client.controlMIT({1: {'q': 0.5, 'kp': 10, 'kd': 1}})
```

### 1.2 TaksClient — 高级客户端

```python
client = TaksClient(
    device_type="Semi-Taks-T1",  # 设备型号
    host="192.168.5.12",         # SDK服务端IP
    port=5555,                    # SDK服务端端口
    use_shm=False,                # 本地共享内存模式
    auto_start_sdk=False,         # 自动启动SDK服务
    with_gripper=True,            # 注册夹爪
    with_imu=False,               # 注册IMU
)
client.connect(wait_data=True, timeout=5.0)  # 连接并等待首帧数据
```

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `client.connect(wait_data, timeout)` | 连接并注册设备 | `bool` |
| `client.controlMIT(joints)` | MIT控制命令 | `bool` |
| `client.get_state()` | 获取关节状态 | `dict[int, JointState]` |
| `client.get_position()` | 获取关节位置 | `dict[int, float]` |
| `client.get_velocity()` | 获取关节速度 | `dict[int, float]` |
| `client.get_torque()` | 获取关节力矩 | `dict[int, float]` |
| `client.get_real_positions()` | 带重试读取真机位置 | `dict[int, float]` |
| `client.send_gripper(left_percent, right_percent, kp, kd)` | 发送夹爪控制 | `None` |
| `client.get_imu()` | 获取IMU数据 | `IMUState` |
| `client.close()` | 断开连接 | `None` |

**属性**:

| 属性 | 说明 | 类型 |
|------|------|------|
| `client.joints` | 已注册关节ID列表 | `list[int]` |
| `client.motor_device` | 底层电机设备 | `TaksDevice` |

### 1.3 JointState — 关节状态

```python
state = client.get_state()   # dict[int, JointState]
js = state[1]
js.pos    # float - 位置 (rad)
js.vel    # float - 速度 (rad/s)
js.tau    # float - 力矩 (Nm)
```

### 1.4 MIT控制命令格式

```python
joints = {
    1: {'q': 0.5, 'dq': 0.0, 'tau': 0.0, 'kp': 10.0, 'kd': 1.0},
    2: {'q': -0.3, 'kp': 10.0, 'kd': 1.0},   # dq/tau可省略(默认0)
}
client.controlMIT(joints)
```

| 字段 | 说明 | 单位 |
|------|------|------|
| `q` | 目标位置 | rad |
| `dq` | 目标速度 | rad/s |
| `tau` | 前馈力矩 | Nm |
| `kp` | 位置增益 | - |
| `kd` | 速度增益 | - |

### 1.5 IMUState — IMU传感器数据

```python
imu_state = client.get_imu()   # 需要 with_imu=True
imu_state.ang_vel    # np.ndarray [wx, wy, wz] (rad/s)
imu_state.lin_acc    # np.ndarray [ax, ay, az] (m/s²)
imu_state.quat       # np.ndarray [w, x, y, z]
imu_state.rpy        # np.ndarray [roll, pitch, yaw] (rad)
```

### 1.6 模块级注册（可选快捷方式）

```python
from taks.client import register

client = register("Semi-Taks-T1", host="192.168.5.12", with_gripper=True)
# 等价于 TaksClient(...).connect(...)
```

---

## 二、taks.ik — 逆运动学

### 2.1 快速开始

```python
from taks.ik import IKController

ik = IKController()

# 主接口：step()一步完成（设置目标+求解+返回MIT命令）
cmd = ik.step(left_pos, left_quat, right_pos, right_quat)

# 或通过末端代理设置目标后调用step()
ik.left_hand.set_target_quat(pos, quat)
cmd = ik.step()
```

### 2.2 IKController

```python
ik = IKController(
    device_type="Semi-Taks-T1",  # 设备型号
    feedforward_scale=1.00,       # 前馈力矩缩放系数
)
```

**核心方法**:

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `ik.step(left_pos, left_quat, right_pos, right_quat, ramp_progress)` | **主接口**：设置目标+求解+返回MIT命令 | `dict[int, dict]` |
| `ik.solve(ramp_progress)` | 仅求解IK，返回MIT命令缓存 | `dict[int, dict]` |
| `ik.set_gains(kp_dict, kd_dict)` | 设置关节增益 | `None` |
| `ik.reset()` | 复位到home位置 | `None` |
| `ik.unfreeze()` | 解冻软保护并复位 | `None` |

**属性**:

| 属性 | 说明 | 类型 |
|------|------|------|
| `ik.left_hand` | 左手末端代理 | `EndEffectorProxy` |
| `ik.right_hand` | 右手末端代理 | `EndEffectorProxy` |
| `ik.joint_positions` | 最近一次solve的关节位置 | `dict[int, float]` |
| `ik.joint_torques` | 最近一次solve的关节力矩 | `dict[int, float]` |
| `ik.is_resetting` | 是否正在复位动画 | `bool` |
| `ik.is_frozen` | 是否被软保护冻结 | `bool` |
| `ik.model` | MuJoCo模型 | `MjModel` |
| `ik.data` | MuJoCo数据 | `MjData` |
| `ik.dt` | 控制步长 (s) | `float` |

> **注意**: `ik.solve()` 返回的是内部缓存字典（复用对象），调用者不应长期持有引用。每帧直接使用返回值即可。

### 2.3 EndEffectorProxy — 末端代理

```python
# 获取当前末端状态
pos, quat = ik.left_hand.eef()         # 位置+四元数
pos  = ik.left_hand.pos()              # 仅位置 np.ndarray[3]
quat = ik.left_hand.quat()             # 仅四元数 np.ndarray[4] [w,x,y,z]
euler = ik.left_hand.euler()           # 欧拉角 [roll, pitch, yaw] (rad)

# 获取home状态
home_pos  = ik.left_hand.home_pos()    # np.ndarray[3]
home_quat = ik.left_hand.home_quat()   # np.ndarray[4]
home_euler = ik.left_hand.home_euler() # [roll, pitch, yaw] (rad)

# 设置末端目标（四元数）
ik.left_hand.set_target_quat(position, quaternion)
ik.left_hand.set_target_quat(position)  # 省略quat则使用home姿态

# 设置末端目标（欧拉角）
ik.left_hand.set_target_euler(position, roll, pitch, yaw)
ik.left_hand.set_target_euler(position, roll=0.1)  # 部分指定，其余用home值
```

### 2.4 HalfBodyIKSolver — 底层求解器

一般不直接使用，`IKController` 内部已封装。如需底层控制：

```python
from taks.ik import HalfBodyIKSolver
import numpy as np

solver = HalfBodyIKSolver()
solver.set_end_effector_target("left_hand", position, quaternion)
solver.set_end_effector_target("right_hand", position, quaternion)
joint_pos = solver.solve(ramp_progress=1.0)   # dict[int, float]
torques = solver.get_joint_torques()           # dict[int, float]
pos, quat = solver.get_end_effector_position("left_hand")
solver.reset()
```

### 2.5 四元数与欧拉角转换

```python
from taks.ik import quat_from_euler, euler_from_quat
import numpy as np

# 欧拉角→四元数 (XYZ顺序，单位rad)
quat = quat_from_euler(roll, pitch, yaw)   # [w, x, y, z]

# 四元数→欧拉角
euler = euler_from_quat(np.array([1.0, 0.0, 0.0, 0.0]))  # [roll, pitch, yaw]
```

### 2.6 模块级注册

```python
import taks.ik

ik = taks.ik.register("Semi-Taks-T1", feedforward_scale=1.00)
_ik = taks.ik.get_instance()  # 获取已注册的实例
```

---

## 三、taks.vr — VR遥操作

### 3.1 快速开始

```python
from taks.vr import VRController
from taks.ik import IKController

ik = IKController()
vr = VRController()
vr.start()

targets = vr.step(ik)           # 返回末端目标字典（可直接 **targets 传给ik.step）
cmd = ik.step(**targets)        # IK求解
left_grip, right_grip = vr.gripper  # 夹爪值

vr.stop()   # 停止VR接收
```

### 3.2 VRController — VR高级控制器

```python
vr = VRController(
    device_type="Semi-Taks-T1",  # 设备型号
    ip="0.0.0.0",                 # 监听IP
    port=7000,                    # 监听端口
    convert_quat=True,            # Unity→MuJoCo四元数转换
)
```

**核心方法**:

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `vr.step(ik)` | **主接口**：处理按钮+偏移+返回末端目标 | `dict` |
| `vr.start()` | 启动VR接收 | `None` |
| `vr.stop()` | 停止VR接收 | `None` |
| `vr.init_offset(ik)` | 手动初始化偏移 | `bool` |
| `vr.reset_offset()` | 重置偏移（触发重新初始化） | `None` |

**属性**:

| 属性 | 说明 | 类型 |
|------|------|------|
| `vr.data` | VR数据快照 | `VRData` |
| `vr.gripper` | 夹爪值 (left, right) | `tuple[float, float]` |
| `vr.tracking_enabled` | 追踪状态 | `bool` |

支持 `with` 语法（自动调用 `start`/`stop`）：

```python
with VRController() as vr:
    targets = vr.step(ik)
```

### 3.3 VRReceiver — 底层接收器

```python
from taks.vr import VRReceiver

receiver = VRReceiver(port=7000)
receiver.start()
data = receiver.data   # VRData
receiver.stop()
```

### 3.4 VRData — 数据结构

```python
data = vr.data

data.head.position          # np.ndarray [x, y, z]
data.head.quaternion         # np.ndarray [w, x, y, z]
data.left_hand.position      # np.ndarray [x, y, z]
data.left_hand.quaternion    # np.ndarray [w, x, y, z]
data.left_hand.gripper       # float 0.0~1.0
data.right_hand.position
data.right_hand.quaternion
data.right_hand.gripper
data.tracking_enabled        # bool
data.timestamp               # float

# 按钮事件（消费型：读取后自动清除）
data.button_events.left_x    # bool
data.button_events.left_y    # bool
data.button_events.right_a   # bool
data.button_events.right_b   # bool
```

### 3.5 四元数工具

```python
from taks.vr import quat_conj, quat_mul, quat_norm

q_inv    = quat_conj(q)           # 共轭 [w, -x, -y, -z]
q_result = quat_mul(q1, q2)       # 乘法
q_unit   = quat_norm(q)           # 归一化
```

### 3.6 VROffset — 偏移管理

```python
from taks.vr import VROffset

offset = VROffset()
offset.left           # np.ndarray - 左手位置偏移
offset.right          # np.ndarray - 右手位置偏移
offset.left_quat      # np.ndarray - 左手姿态偏移（四元数）
offset.right_quat     # np.ndarray - 右手姿态偏移（四元数）
offset.initialized    # bool
```

### 3.7 模块级注册

```python
import taks.vr

vr = taks.vr.register("Semi-Taks-T1", port=7000)  # 自动启动
vr2 = taks.vr.get_instance()  # 获取已注册的实例
```

### 3.8 CLI工具

```bash
python -m taks.vr --port 7000 --hz 10
```

---

## 四、完整示例

### 4.1 VR遥操作（async/uvloop版）

```python
import asyncio
try:
    import uvloop; uvloop.install()
except ImportError:
    pass
from contextlib import asynccontextmanager
from taks.client import TaksClient
from taks.ik import IKController
from taks.vr import VRController

@asynccontextmanager
async def managed_client(host):
    client = TaksClient("Semi-Taks-T1", host=host, with_gripper=True)
    client.connect(wait_data=True)
    try:
        yield client
    finally:
        client.close()

async def main():
    ik = IKController()
    vr = VRController()
    vr.start()
    async with managed_client("192.168.5.12") as client:
        while True:
            targets = vr.step(ik)
            cmd = ik.step(**targets)
            client.controlMIT(cmd)
            client.send_gripper(*vr.gripper)
            await asyncio.sleep(0.02)

asyncio.run(main())
```

### 4.2 读取真机数据

```python
from taks.client import TaksClient

client = TaksClient("Semi-Taks-T1", host="192.168.5.12")
client.connect(wait_data=True)
state = client.get_state()
for jid, js in sorted(state.items()):
    print(f"  关节{jid}: pos={js.pos:.4f} vel={js.vel:.4f} tau={js.tau:.4f}")
client.close()
```

### 4.3 键盘控制IK（纯仿真）

```bash
python example/demo_ik_keyboard.py --step 0.005 --freq 100
```

### 4.4 末端画圈

```bash
python example/demo_ik_circle.py --radius 0.06 --period 4.0
```

### 4.5 缓启动/缓停止

```bash
python example/demo_ramp_up.py
python example/demo_ramp_down.py
```

---

## 五、关节ID映射

### Semi-Taks-T1（22关节）

| 部位 | 关节名 | ID |
|------|--------|----|
| 右臂 | right_shoulder_pitch | 1 |
| 右臂 | right_shoulder_roll | 2 |
| 右臂 | right_shoulder_yaw | 3 |
| 右臂 | right_elbow | 4 |
| 右臂 | right_wrist_roll | 5 |
| 右臂 | right_wrist_yaw | 6 |
| 右臂 | right_wrist_pitch | 7 |
| 右手夹爪 | right_gripper | 8 |
| 左臂 | left_shoulder_pitch | 9 |
| 左臂 | left_shoulder_roll | 10 |
| 左臂 | left_shoulder_yaw | 11 |
| 左臂 | left_elbow | 12 |
| 左臂 | left_wrist_roll | 13 |
| 左臂 | left_wrist_yaw | 14 |
| 左臂 | left_wrist_pitch | 15 |
| 左手夹爪 | left_gripper | 16 |
| 腰 | waist_yaw | 17 |
| 腰 | waist_roll | 18 |
| 腰 | waist_pitch | 19 |
| 颈 | neck_yaw | 20 |
| 颈 | neck_roll | 21 |
| 颈 | neck_pitch | 22 |

### Taks-T1（34关节）

在 Semi-Taks-T1 基础上增加双腿关节 23-34。

---

## 六、通信协议

SDK 使用 [Zenoh](https://zenoh.io/) 进行客户端-服务端通信。

- **传输模式**: QUIC（远程）或 SHM 共享内存（本地）
- **状态话题**: `taks/state/motor`、`taks/state/imu`
- **命令话题**: `taks/cmd/{device}/{command}`
- **编码格式**: orjson（二进制JSON）

---

## 七、常见问题

**Q: 如何在仿真中测试IK？**
A: `taks.ik` 不依赖真机，直接使用即可。参考 `example/demo_ik_keyboard.py`。

**Q: VR接收器支持哪些后端？**
A: 优先使用 C++ 编译库（高性能），不可用时自动回退到纯 Python 实现。

**Q: 为什么连接失败？**
A: 确认 SDK 服务端已启动，IP/端口正确，且网络连通。可用 `auto_start_sdk=True` 自动启动本地服务。

**Q: 如何自定义MuJoCo模型路径？**
A: `HalfBodyIKSolver(xml_path=Path("/path/to/model.xml"))`
