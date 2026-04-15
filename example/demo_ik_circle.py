#!/usr/bin/env python3
"""左右手画圈演示 - X轴不动

左右手在YZ平面上画圆，X轴保持不变。

使用方法:
  python demo_ik_circle.py
  python demo_ik_circle.py --radius 0.05 --period 3.0
"""

import sys
import os
import signal
import argparse
import math

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

from taks.ik import IKController


def main():
    parser = argparse.ArgumentParser(description="左右手画圈演示")
    parser.add_argument("--radius", type=float, default=0.06, help="圆半径(m)")
    parser.add_argument("--period", type=float, default=4.0, help="周期(s)")
    parser.add_argument("--freq", type=float, default=50.0, help="控制频率(Hz)")
    parser.add_argument("--opposite", action="store_true", help="左右手反向旋转")
    args = parser.parse_args()

    # 初始化IK
    ik = IKController()
    rate = RateLimiter(frequency=args.freq, warn=False)
    ik.dt = rate.dt

    # 获取home位置作为圆心
    left_center = ik.left_hand.home_pos().copy()
    right_center = ik.right_hand.home_pos().copy()
    left_quat = ik.left_hand.home_quat()
    right_quat = ik.right_hand.home_quat()

    # 信号处理
    running = [True]

    def sig_handler(sig, frame):
        running[0] = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("=" * 50)
    print("  左右手画圈演示 - X轴不动")
    print("=" * 50)
    print(f"  半径: {args.radius}m  周期: {args.period}s")
    print(f"  旋转方向: {'反向' if args.opposite else '同向'}")
    print("  按 Backspace 复位")
    print("=" * 50)

    def key_callback(keycode):
        if keycode == 259:  # Backspace
            ik.reset()
            print("[复位] 回home位置")

    # 主循环
    with mujoco.viewer.launch_passive(
        ik.model,
        ik.data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(ik.model, viewer.cam)
        t = 0.0
        frame_count = 0

        while viewer.is_running() and running[0]:
            if not ik.is_resetting:
                # 计算圆周运动角度
                angle = 2 * math.pi * t / args.period

                # 左手：YZ平面画圆（X不变）
                left_pos = left_center.copy()
                left_pos[1] += args.radius * math.cos(angle)  # X
                left_pos[2] += args.radius * math.sin(angle)  # Z

                # 右手：YZ平面画圆（X不变）
                right_angle = -angle if args.opposite else angle
                right_pos = right_center.copy()
                right_pos[1] += args.radius * math.cos(right_angle)  # X
                right_pos[2] += args.radius * math.sin(right_angle)  # Z

                # IK求解
                ik.step(left_pos, left_quat, right_pos, right_quat)
            else:
                ik.step()

            # 渲染
            mujoco.mj_camlight(ik.model, ik.data)
            viewer.sync()

            # 打印状态
            frame_count += 1
            if frame_count % 50 == 0:
                lp = ik.left_hand.pos()
                rp = ik.right_hand.pos()
                print(
                    f"t={t:.1f}s | [L] x={lp[0]:.3f} z={lp[2]:.3f} | [R] x={rp[0]:.3f} z={rp[2]:.3f}"
                )

            t += rate.dt
            rate.sleep()

    print("[完成] 程序退出")


if __name__ == "__main__":
    main()
