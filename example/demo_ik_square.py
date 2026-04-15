#!/usr/bin/env python3
"""左右手在YZ平面画正方形演示 - X轴不动

左右手以home位置为中心，在YZ平面上画正方形，X轴保持不变。

使用方法:
  python demo_ik_square.py
  python demo_ik_square.py --side 0.08 --period 4.0
"""

import sys
import os
import signal
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

from taks.ik import IKController


def square_point(t, period, side):
    """根据时间t计算正方形轨迹上的(y_offset, z_offset)

    正方形按顺序经过四条边：上→右→下→左，以中心为原点，边长为side。
    """
    half = side / 2.0
    # 归一化到[0, 1)
    phase = (t % period) / period
    # 四段各占1/4周期
    seg = phase * 4.0
    seg_idx = int(seg)
    seg_frac = seg - seg_idx  # 当前段内进度[0,1)

    if seg_idx == 0:
        # 上边：y从-half到+half，z=+half
        y = -half + seg_frac * side
        z = half
    elif seg_idx == 1:
        # 右边：y=+half，z从+half到-half
        y = half
        z = half - seg_frac * side
    elif seg_idx == 2:
        # 下边：y从+half到-half，z=-half
        y = half - seg_frac * side
        z = -half
    else:
        # 左边：y=-half，z从-half到+half
        y = -half
        z = -half + seg_frac * side

    return y, z


def main():
    parser = argparse.ArgumentParser(description="左右手在YZ平面画正方形演示")
    parser.add_argument("--side", type=float, default=0.08, help="正方形边长(m)")
    parser.add_argument("--period", type=float, default=4.0, help="单圈周期(s)")
    parser.add_argument("--freq", type=float, default=50.0, help="控制频率(Hz)")
    parser.add_argument("--opposite", action="store_true", help="左右手反向运动")
    args = parser.parse_args()

    # 初始化IK
    ik = IKController()
    rate = RateLimiter(frequency=args.freq, warn=False)
    ik.dt = rate.dt

    # 以home位置为正方形中心
    left_center = ik.left_hand.home_pos().copy()
    right_center = ik.right_hand.home_pos().copy()
    left_quat = ik.left_hand.home_quat()
    right_quat = ik.right_hand.home_quat()

    running = [True]

    def sig_handler(sig, frame):
        running[0] = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("=" * 50)
    print("  左右手画正方形演示 - YZ平面，X轴不动")
    print("=" * 50)
    print(f"  边长: {args.side}m  周期: {args.period}s")
    print(f"  方向: {'左右反向' if args.opposite else '同向'}")
    print("  按 Backspace 复位")
    print("=" * 50)

    def key_callback(keycode):
        if keycode == 259:  # Backspace
            ik.reset()
            print("[复位] 回home位置")

    t = 0.0
    frame_count = 0

    with mujoco.viewer.launch_passive(
        ik.model,
        ik.data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(ik.model, viewer.cam)

        while viewer.is_running() and running[0]:
            if not ik.is_resetting:
                # 左手正方形轨迹偏移
                ly, lz = square_point(t, args.period, args.side)
                left_pos = left_center.copy()
                left_pos[1] += ly  # Y轴偏移
                left_pos[2] += lz  # Z轴偏移
                # X轴保持不变

                # 右手：可同向或反向
                rt = (
                    (args.period - t % args.period) % args.period
                    if args.opposite
                    else t
                )
                ry, rz = square_point(rt, args.period, args.side)
                right_pos = right_center.copy()
                right_pos[1] += ry
                right_pos[2] += rz

                ik.step(left_pos, left_quat, right_pos, right_quat)
            else:
                ik.step()

            mujoco.mj_camlight(ik.model, ik.data)
            viewer.sync()

            frame_count += 1
            if frame_count % 50 == 0:
                lp = ik.left_hand.pos()
                rp = ik.right_hand.pos()
                print(
                    f"t={t:.1f}s | "
                    f"[L] y={lp[1]:.3f} z={lp[2]:.3f} | "
                    f"[R] y={rp[1]:.3f} z={rp[2]:.3f}"
                )

            t += rate.dt
            rate.sleep()

    print("[完成] 程序退出")


if __name__ == "__main__":
    main()
