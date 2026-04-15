#!/usr/bin/env python3
"""VR遥操作IK演示

使用方法:
  python example/demo_ik_vr.py
"""

import sys
import os
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

from taks.ik import IKController
from taks.vr import VRController

FREQ = 50.0  # 控制频率(Hz)


def main():
    ik = IKController()
    vr = VRController()
    vr.start()
    rate = RateLimiter(frequency=FREQ, warn=False)
    ik.dt = rate.dt

    print("=" * 50)
    print("  VR遥操作IK演示")
    print("  VR手柄: A=复位  B=解冻/复位")
    print("  键盘: Backspace=复位")
    print("=" * 50)

    def key_callback(keycode):
        if keycode == 259:  # Backspace
            if ik.is_frozen:
                ik.unfreeze()
                vr.reset_offset()
            else:
                ik.reset()
                vr.reset_offset()

    with mujoco.viewer.launch_passive(
        ik.model,
        ik.data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(ik.model, viewer.cam)
        last_print = time.monotonic()

        while viewer.is_running():
            targets = vr.step(ik)   # VR数据→末端目标
            ik.step(**targets)       # 末端目标→关节角

            viewer.sync()
            rate.sleep()

            # 每秒打印一次状态
            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                left_pos = ik.left_hand.pos()
                right_pos = ik.right_hand.pos()
                left_grip, right_grip = vr.gripper
                print(
                    f"[L] pos=[{left_pos[0]:.3f},{left_pos[1]:.3f},{left_pos[2]:.3f}] grip={left_grip:.2f} | "
                    f"[R] pos=[{right_pos[0]:.3f},{right_pos[1]:.3f},{right_pos[2]:.3f}] grip={right_grip:.2f}"
                )

    vr.stop()
    print("[完成] 程序退出")


if __name__ == "__main__":
    main()
