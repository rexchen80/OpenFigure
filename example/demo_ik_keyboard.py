#!/usr/bin/env python3
"""小键盘控制IK演示 - 纯MuJoCo仿真，无需真机

用小键盘数字键控制末端执行器位置和姿态，Tab切换左手/右手模式。

操作说明（小键盘）:
  位置控制:
    8/2 - 前/后 (X轴)
    4/6 - 左/右 (Y轴)
    9/3 - 上/下 (Z轴)
  姿态控制:
    7/1 - Roll  +/-
    +/- - Pitch +/-
    //Enter - Yaw +/-
  功能键:
    Tab   - 切换左手/右手模式
    0     - 复位到home
    .     - 解冻软保护
    Ctrl+C - 退出

使用方法:
  python demo_ik_keyboard.py
  python demo_ik_keyboard.py --step 0.005 --rot 0.05
"""

import sys
import os
import signal
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

from taks.ik import IKController

# 小键盘keycode（MuJoCo）
KP_0 = 320  # 复位
KP_1 = 321  # Roll-
KP_2 = 322  # Y- (后)
KP_3 = 323  # Z- (下)
KP_4 = 324  # X- (左)
KP_5 = 325  # (未用)
KP_6 = 326  # X+ (右)
KP_7 = 327  # Roll+
KP_8 = 328  # Y+ (前)
KP_9 = 329  # Z+ (上)
KP_DOT = 330  # 解冻
KP_DIV = 331  # Yaw-
KP_MUL = 332  # (未用)
KP_SUB = 333  # Pitch-
KP_ADD = 334  # Pitch+
KP_ENTER = 335  # Yaw+
KEY_TAB = 258  # 切换模式
KEY_BACKSPACE = 259

# 模式
MODE_LEFT = 0
MODE_RIGHT = 1
MODE_NAMES = ["左手", "右手"]


class KeyboardIKDemo:
    def __init__(
        self, pos_step: float = 0.005, rot_step: float = 0.03, freq: float = 50.0
    ):
        self.pos_step = pos_step
        self.rot_step = rot_step
        self.freq = freq
        self.ik = IKController(device_type="Semi-Taks-T1")
        self.ik.dt = 1.0 / freq
        self.model = self.ik.model
        self.data = self.ik.data
        self.rate = RateLimiter(frequency=freq, warn=False)
        self.mode = MODE_LEFT
        self._running = True
        # 末端目标
        self._left_pos = self.ik.left_hand.home_pos().astype(np.float32)
        self._right_pos = self.ik.right_hand.home_pos().astype(np.float32)
        self._left_euler = self.ik.left_hand.home_euler()
        self._right_euler = self.ik.right_hand.home_euler()
        self._pressed = set()
        self._frame_count = 0

    def _on_key(self, keycode):
        """键盘回调"""
        if keycode == KEY_TAB:
            self.mode = 1 - self.mode
            print(f"[模式] {MODE_NAMES[self.mode]}")
        elif keycode == KP_0:
            self._reset()
        elif keycode == KP_DOT or keycode == KEY_BACKSPACE:
            if self.ik.is_frozen:
                self.ik.unfreeze()
                self._reset_targets()
                print("[解冻] 软保护已解除")
            else:
                self._reset()
        else:
            self._pressed.add(keycode)

    def _reset_targets(self):
        """重置目标到home"""
        self._left_pos = self.ik.left_hand.home_pos().astype(np.float32)
        self._right_pos = self.ik.right_hand.home_pos().astype(np.float32)
        self._left_euler = self.ik.left_hand.home_euler()
        self._right_euler = self.ik.right_hand.home_euler()

    def _reset(self):
        """复位IK"""
        self.ik.reset()
        self._reset_targets()
        print("[复位] 回home位置")

    def _process_keys(self):
        """处理按键输入"""
        ps, rs = self.pos_step, self.rot_step
        dp = np.zeros(3, dtype=np.float32)
        dr = np.zeros(3, dtype=np.float32)

        # 位置: 8/2=前后, 4/6=左右, 9/3=上下
        if KP_8 in self._pressed:
            dp[0] += ps
        if KP_2 in self._pressed:
            dp[0] -= ps
        if KP_4 in self._pressed:
            dp[1] += ps
        if KP_6 in self._pressed:
            dp[1] -= ps
        if KP_9 in self._pressed:
            dp[2] += ps
        if KP_3 in self._pressed:
            dp[2] -= ps

        # 姿态: 7/1=Roll, +/-=Pitch, //Enter=Yaw
        if KP_7 in self._pressed:
            dr[0] += rs
        if KP_1 in self._pressed:
            dr[0] -= rs
        if KP_ADD in self._pressed:
            dr[1] += rs
        if KP_SUB in self._pressed:
            dr[1] -= rs
        if KP_ENTER in self._pressed:
            dr[2] += rs
        if KP_DIV in self._pressed:
            dr[2] -= rs

        # 应用到当前模式
        if self.mode == MODE_LEFT:
            self._left_pos += dp
            self._left_euler += dr
        else:
            self._right_pos += dp
            self._right_euler += dr

        self._pressed.clear()

    def step(self):
        """单步控制"""
        self._process_keys()
        if not self.ik.is_resetting:
            self.ik.left_hand.set_target_euler(self._left_pos, *self._left_euler)
            self.ik.right_hand.set_target_euler(self._right_pos, *self._right_euler)
        mit_cmd = self.ik.step()  # 直接返回MIT命令
        self._frame_count += 1
        if self._frame_count % int(self.freq) == 0:
            lp, rp = self._left_pos, self._right_pos
            le, re = self._left_euler, self._right_euler
            hand = "L" if self.mode == MODE_LEFT else "R"
            pos = lp if self.mode == MODE_LEFT else rp
            eul = le if self.mode == MODE_LEFT else re
            print(
                f"[{hand}] pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
                f"rpy=[{np.degrees(eul[0]):.1f},{np.degrees(eul[1]):.1f},{np.degrees(eul[2]):.1f}]°"
            )
        return mit_cmd

    def run(self):
        """启动MuJoCo viewer"""
        print("=" * 55)
        print("  小键盘控制IK演示 - MuJoCo仿真")
        print("=" * 55)
        print("  位置: 8/2=前后  4/6=左右  9/3=上下")
        print("  姿态: 7/1=Roll  +/-=Pitch  //Enter=Yaw")
        print("  Tab=切换左/右手  0=复位  .=解冻")
        print("=" * 55)
        print(f"  当前模式: {MODE_NAMES[self.mode]}")
        print("=" * 55)

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self._on_key,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            while viewer.is_running() and self._running:
                self.step()
                mujoco.mj_camlight(self.model, self.data)
                viewer.sync()
                self.rate.sleep()

    def close(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="小键盘控制IK演示")
    parser.add_argument("--step", type=float, default=0.005, help="位置步长(m)")
    parser.add_argument("--rot", type=float, default=0.03, help="姿态步长(rad)")
    parser.add_argument("--freq", type=float, default=50.0, help="控制频率(Hz)")
    args = parser.parse_args()

    demo = KeyboardIKDemo(pos_step=args.step, rot_step=args.rot, freq=args.freq)
    _shutting_down = False

    def sig_handler(sig, frame):
        nonlocal _shutting_down
        if _shutting_down:
            os._exit(1)
        _shutting_down = True
        print("\n[退出] 安全关闭...")
        demo.close()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        demo.run()
    except Exception as e:
        print(f"[错误] {e}")
    finally:
        print("[完成] 程序退出")


if __name__ == "__main__":
    main()
