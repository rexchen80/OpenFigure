import argparse
import asyncio
import os
import socket
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import orjson
import uvloop
from rich.console import Console
from rich.live import Live
from rich.table import Table

UDP_IP = "0.0.0.0"
UDP_PORT = 7000
_USE_CPP = False
_CppVRReceiver = None
try:
    _curr = os.path.abspath(__file__)
    while _curr != "/" and os.path.basename(_curr) != "backend":
        _curr = os.path.dirname(_curr)
    _lib_path = os.path.join(_curr, "libs", "drivers", "libs")
    if _lib_path not in sys.path:
        sys.path.insert(0, _lib_path)
    from dm_can.vr import VRReceiver as _CppVRReceiver

    _USE_CPP = True
except ImportError:
    pass


@dataclass
class VRPose:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    gripper: float = 0.0


@dataclass
class VRButtonEvents:
    left_x: bool = False
    left_y: bool = False
    right_a: bool = False
    right_b: bool = False


@dataclass
class VRData:
    head: VRPose = field(default_factory=VRPose)
    left_hand: VRPose = field(default_factory=VRPose)
    right_hand: VRPose = field(default_factory=VRPose)
    tracking_enabled: bool = False
    timestamp: float = 0.0
    total_offset: float = 0.0
    button_events: VRButtonEvents = field(default_factory=VRButtonEvents)


def quat_conj(q):
    result = q.copy()
    result[1:] *= -1.0
    return result


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    result = np.empty(4, dtype=np.float64)
    result[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    result[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    result[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    result[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return result


def quat_norm(q):
    sq_norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    if sq_norm < np.float64(1e-16):
        identity = np.empty(4, dtype=np.float64)
        identity[0] = 1.0
        identity[1] = identity[2] = identity[3] = 0.0
        return identity
    return q * (np.float64(1.0) / np.sqrt(np.float64(sq_norm)))


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1", "yes", "on"}:
            return True
        if norm in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _copy_pose(src_pose, dst_pose):
    dst_pose.position[:] = src_pose.position
    dst_pose.quaternion[:] = src_pose.quaternion
    dst_pose.gripper = src_pose.gripper


def _copy_vr_data(src, dst):
    _copy_pose(src.head, dst.head)
    _copy_pose(src.left_hand, dst.left_hand)
    _copy_pose(src.right_hand, dst.right_hand)
    dst.tracking_enabled = src.tracking_enabled
    dst.timestamp = src.timestamp
    dst.total_offset = src.total_offset


@dataclass
class VROffset:
    left: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    right: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    left_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    right_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    initialized: bool = False


class _CppBackend:
    __slots__ = "_cpp", "_snap"

    def __init__(self, ip, port, convert_quat):
        assert _CppVRReceiver is not None
        self._cpp = _CppVRReceiver(ip, port, convert_quat)
        self._snap = VRData()

    @staticmethod
    def _copy_pose(src_pose, dst_pose):
        pos = src_pose.position
        dst_pose.position[0] = pos.x
        dst_pose.position[1] = pos.y
        dst_pose.position[2] = pos.z
        quat = src_pose.quaternion
        dst_pose.quaternion[0] = quat.w
        dst_pose.quaternion[1] = quat.x
        dst_pose.quaternion[2] = quat.y
        dst_pose.quaternion[3] = quat.z
        dst_pose.gripper = float(src_pose.gripper)

    @property
    def data(self):
        raw = self._cpp.get_data()
        snap = self._snap
        self._copy_pose(raw.head, snap.head)
        self._copy_pose(raw.left_hand, snap.left_hand)
        self._copy_pose(raw.right_hand, snap.right_hand)
        snap.tracking_enabled = bool(raw.tracking_enabled)
        snap.timestamp = float(raw.timestamp)
        snap.total_offset = float(raw.total_offset)
        btn = raw.button_events
        snap.button_events.left_x = bool(btn.left_x)
        snap.button_events.left_y = bool(btn.left_y)
        snap.button_events.right_a = bool(btn.right_a)
        snap.button_events.right_b = bool(btn.right_b)
        return snap

    @property
    def tracking_enabled(self):
        return self._cpp.tracking_enabled()

    def reset_smooth(self):
        self._cpp.reset_smooth()

    def start(self):
        self._cpp.start()

    def stop(self):
        self._cpp.stop()


class _VRProtocol(asyncio.DatagramProtocol):
    def __init__(self, backend):
        self._backend = backend

    def datagram_received(self, data, addr):
        if data:
            self._backend._process_packet(data)


class _PyBackend:
    BTN_DEBOUNCE_SEC = 0.05
    BTN_NAMES = "leftX", "leftY", "rightA", "rightB"
    BTN_LEGACY_NAMES = "left_x", "left_y", "right_a", "right_b"

    def __init__(self, ip, port, convert_quat):
        self._ip, self._port = ip, port
        self._convert_quat = convert_quat
        self._running = False
        self._loop = None
        self._loop_thread = None
        self._transport = None
        self._initialized = False
        self._btn_times = [0.0] * 4
        self._btn_pending = [0] * 4
        import threading as _threading

        self._bufs = [VRData(), VRData()]
        self._read_idx = 1
        self._snap = VRData()
        self._swap_lock = _threading.Lock()
        self._stop_event = None
        self._recv_ts = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._loop = uvloop.new_event_loop()
        import threading as _threading

        self._loop_thread = _threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        print(f"[VR] listening on {self._ip}:{self._port} (Python/uvloop)")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        self._stop_event = asyncio.Event()
        (
            self._transport,
            _proto,
        ) = await self._loop.create_datagram_endpoint(
            lambda: _VRProtocol(self),
            local_addr=(self._ip, self._port),
            reuse_port=True,
        )
        sock = self._transport.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 524288)
        await self._stop_event.wait()
        self._transport.close()

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running() and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)
        if self._loop and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
        self._loop_thread = None
        self._transport = None
        self._stop_event = None

    @property
    def data(self):
        with self._swap_lock:
            src = self._bufs[self._read_idx]
            snap = self._snap
            _copy_vr_data(src, snap)
        snap.button_events.left_x = self._consume_button(0)
        snap.button_events.left_y = self._consume_button(1)
        snap.button_events.right_a = self._consume_button(2)
        snap.button_events.right_b = self._consume_button(3)
        return snap

    @property
    def tracking_enabled(self):
        with self._swap_lock:
            return self._bufs[self._read_idx].tracking_enabled

    def reset_smooth(self):
        self._initialized = False

    def _consume_button(self, idx):
        with self._swap_lock:
            pending = self._btn_pending[idx]
            if pending <= 0:
                return False
            self._btn_pending[idx] = pending - 1
            return True

    def _set_button(self, idx, value):
        if idx < 0 or idx >= 4:
            return
        if not value:
            return
        ts = time.monotonic()
        if ts - self._btn_times[idx] > self.BTN_DEBOUNCE_SEC:
            self._btn_times[idx] = ts
            self._btn_pending[idx] += 1

    def _process_packet(self, raw):
        # 逐帧解析
        frames = []
        raw_buf = raw if isinstance(raw, (bytes, memoryview)) else raw.encode()
        offset = 0
        while offset < len(raw_buf):
            # 跳过帧间空白
            while offset < len(raw_buf) and raw_buf[offset] <= 0x20:
                offset += 1
            if offset >= len(raw_buf):
                break
            try:
                obj = orjson.loads(raw_buf[offset:])
                frames.append(obj)
                break
            except orjson.JSONDecodeError:
                # 尝试找到第一个完整帧：用括号深度扫描
                depth = 0
                in_str = False
                esc = False
                i = offset
                frame_end = -1
                while i < len(raw_buf):
                    c = raw_buf[i]
                    if esc:
                        esc = False
                        i += 1
                        continue
                    if c == 0x5C and in_str:  # '\'
                        esc = True
                        i += 1
                        continue
                    if c == 0x22:  # '"'
                        in_str = not in_str
                        i += 1
                        continue
                    if in_str:
                        i += 1
                        continue
                    if c == 0x7B:  # '{'
                        depth += 1
                    elif c == 0x7D:  # '}'
                        depth -= 1
                        if depth == 0:
                            frame_end = i + 1
                            break
                    i += 1
                if frame_end < 0:
                    break
                try:
                    frames.append(orjson.loads(raw_buf[offset:frame_end]))
                except Exception:
                    pass
                offset = frame_end
        if not frames:
            return
        # 全部帧提取按钮事件，最后一帧作为位姿数据
        packet = frames[-1]
        with self._swap_lock:
            write_idx = 1 - self._read_idx
            src = self._bufs[self._read_idx]
            buf = self._bufs[write_idx]
            _copy_vr_data(src, buf)
            packet_timestamp = packet.get("timestamp")
            event_ts = float(packet_timestamp) if packet_timestamp is not None else None
            self._parse_pose(packet.get("head"), buf.head)
            self._parse_pose(packet.get("leftHand"), buf.left_hand)
            self._parse_pose(packet.get("rightHand"), buf.right_hand)
            tracking = packet.get("trackingEnabled")
            if tracking is not None:
                buf.tracking_enabled = _coerce_bool(tracking)
            elif "tracking_enabled" in packet:
                buf.tracking_enabled = _coerce_bool(packet["tracking_enabled"])
            if event_ts is not None:
                buf.timestamp = event_ts
            offset_val = packet.get("totalOffset")
            if offset_val is not None:
                buf.total_offset = float(offset_val)
            for frame in frames:
                btn_events = frame.get("buttonEvents") or {}
                for btn_idx, btn_name in enumerate(self.BTN_NAMES):
                    val = btn_events.get(btn_name)
                    if val is None:
                        val = btn_events.get(self.BTN_LEGACY_NAMES[btn_idx])
                    if val is not None:
                        self._set_button(btn_idx, _coerce_bool(val))
            self._read_idx = write_idx
        self._initialized = True
        self._recv_ts = time.monotonic()

    def _parse_pose(self, obj, pose):
        if not obj:
            return
        pos_data = obj.get("position")
        if pos_data and isinstance(pos_data, list) and len(pos_data) >= 3:
            pose.position[0] = pos_data[0]
            pose.position[1] = pos_data[1]
            pose.position[2] = pos_data[2]
        quat_data = obj.get("quaternion")
        if quat_data and isinstance(quat_data, list) and len(quat_data) >= 4:
            if self._convert_quat:
                pose.quaternion[0] = float(quat_data[0])
                pose.quaternion[1] = -float(quat_data[1])
                pose.quaternion[2] = -float(quat_data[2])
                pose.quaternion[3] = -float(quat_data[3])
            else:
                pose.quaternion[0] = quat_data[0]
                pose.quaternion[1] = quat_data[1]
                pose.quaternion[2] = quat_data[2]
                pose.quaternion[3] = quat_data[3]
        if "gripper" in obj:
            pose.gripper = float(obj["gripper"])


class VRReceiver:
    def __init__(self, ip=UDP_IP, port=UDP_PORT, convert_quat=True):
        self.ip, self.port = ip, port
        self.convert_quat = convert_quat
        if _USE_CPP:
            self._backend = _CppBackend(ip, port, convert_quat)
        else:
            self._backend = _PyBackend(ip, port, convert_quat)
        self._backend_type = "C++" if _USE_CPP else "Python"

    @property
    def data(self):
        return self._backend.data

    @property
    def tracking_enabled(self):
        return self._backend.tracking_enabled

    def reset_smooth(self):
        self._backend.reset_smooth()

    def start(self):
        self._backend.start()

    def stop(self):
        self._backend.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class VRController:
    def __init__(
        self,
        device_type="Semi-Taks-T1",
        ip=UDP_IP,
        port=UDP_PORT,
        convert_quat=True,
    ):
        self.device_type = device_type
        self.receiver = VRReceiver(ip=ip, port=port, convert_quat=convert_quat)
        self.offset = VROffset()
        self._last_b_state = False
        self._started = False
        self._ik_ref = None

    def start(self):
        if not self._started:
            self.receiver.start()
            self._started = True

    def stop(self):
        if self._started:
            self.receiver.stop()
            self._started = False

    @property
    def data(self):
        return self.receiver.data

    @property
    def tracking_enabled(self):
        return self.receiver.tracking_enabled

    def init_offset(self, ik):
        vr_data = self.receiver.data
        if not vr_data.tracking_enabled:
            return False
        left_pos = ik.left_hand.mocap_pos()
        right_pos = ik.right_hand.mocap_pos()
        left_quat = ik.left_hand.mocap_quat()
        right_quat = ik.right_hand.mocap_quat()
        self.offset.left = left_pos - vr_data.left_hand.position
        self.offset.right = right_pos - vr_data.right_hand.position
        self.offset.left_quat = quat_norm(
            quat_mul(quat_conj(vr_data.left_hand.quaternion), left_quat)
        )
        self.offset.right_quat = quat_norm(
            quat_mul(
                quat_conj(vr_data.right_hand.quaternion),
                right_quat,
            )
        )
        self.offset.initialized = True
        self.receiver.reset_smooth()
        return True

    def reset_offset(self):
        self.offset.initialized = False

    def step(self, ik=None):
        vr_data = self.receiver.data
        if ik is not None:
            self._ik_ref = ik
            btn_b = vr_data.button_events.right_b
            if btn_b and not self._last_b_state:
                if ik.is_frozen:
                    ik.unfreeze()
                    self.offset.initialized = False
                else:
                    ik.reset()
                    self.offset.initialized = False
            self._last_b_state = btn_b
            if vr_data.button_events.right_a:
                ik.reset()
                self.offset.initialized = False
            if ik.is_resetting:
                return {}
            if not self.offset.initialized and vr_data.tracking_enabled:
                self.init_offset(ik)
        if not self.offset.initialized:
            return {}
        left_quat_out = quat_norm(
            quat_mul(
                vr_data.left_hand.quaternion,
                self.offset.left_quat,
            )
        )
        right_quat_out = quat_norm(
            quat_mul(
                vr_data.right_hand.quaternion,
                self.offset.right_quat,
            )
        )
        return {
            "left_pos": vr_data.left_hand.position + self.offset.left,
            "left_quat": left_quat_out,
            "right_pos": vr_data.right_hand.position + self.offset.right,
            "right_quat": right_quat_out,
        }

    @property
    def gripper(self):
        vr_data = self.receiver.data
        return (
            vr_data.left_hand.gripper,
            vr_data.right_hand.gripper,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


_instance = None


def register(
    device_type="Semi-Taks-T1",
    ip=UDP_IP,
    port=UDP_PORT,
    convert_quat=True,
):
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = VRController(
        device_type=device_type,
        ip=ip,
        port=port,
        convert_quat=convert_quat,
    )
    _instance.start()
    return _instance


def get_instance():
    return _instance


def _fmt_vec(v):
    return f"[{v[0]: .4f}, {v[1]: .4f}, {v[2]: .4f}]"


def _fmt_quat(q):
    return f"[{q[0]: .4f}, {q[1]: .4f}, {q[2]: .4f}, {q[3]: .4f}]"


def _build_table(data, backend_type="Python"):
    table = Table(
        title=f"VR 原始数据  [backend: {backend_type}]",
        show_lines=True,
    )
    table.add_column("字段", style="cyan", no_wrap=True)
    table.add_column("值", style="white")
    table.add_row("tracking_enabled", str(data.tracking_enabled))
    table.add_row("timestamp", f"{data.timestamp:.6f}")
    table.add_row("total_offset", f"{data.total_offset:.6f}")
    table.add_row("head.position", _fmt_vec(data.head.position))
    table.add_row("head.quaternion", _fmt_quat(data.head.quaternion))
    table.add_row("left_hand.position", _fmt_vec(data.left_hand.position))
    table.add_row(
        "left_hand.quaternion",
        _fmt_quat(data.left_hand.quaternion),
    )
    table.add_row("left_hand.gripper", f"{data.left_hand.gripper:.6f}")
    table.add_row("right_hand.position", _fmt_vec(data.right_hand.position))
    table.add_row(
        "right_hand.quaternion",
        _fmt_quat(data.right_hand.quaternion),
    )
    table.add_row("right_hand.gripper", f"{data.right_hand.gripper:.6f}")
    table.add_row("button.left_x", str(data.button_events.left_x))
    table.add_row("button.left_y", str(data.button_events.left_y))
    table.add_row("button.right_a", str(data.button_events.right_a))
    table.add_row("button.right_b", str(data.button_events.right_b))
    return table


def main():
    parser = argparse.ArgumentParser(description="VR UDP接收器原始数据查看")
    parser.add_argument("--ip", type=str, default=UDP_IP)
    parser.add_argument("--port", type=int, default=UDP_PORT)
    parser.add_argument("--hz", type=float, default=1e1)
    parser.add_argument("--no-convert-quat", action="store_true")
    args = parser.parse_args()
    console = Console()
    receiver = VRReceiver(
        ip=args.ip,
        port=args.port,
        convert_quat=not args.no_convert_quat,
    )
    sleep_interval = 1.0 / max(args.hz, 0.1)
    receiver.start()
    console.print(
        f"[green][VR] 启动成功 ({receiver._backend_type})，监听 {args.ip}:{args.port}[/green]"
    )
    console.print("[yellow]按 Ctrl+C 退出[/yellow]")
    try:
        with Live(
            console=console,
            refresh_per_second=max(args.hz, 0.1),
            screen=False,
        ) as live:
            while True:
                live.update(_build_table(receiver.data, receiver._backend_type))
                time.sleep(sleep_interval)
    except KeyboardInterrupt:
        console.print("[yellow]\n[VR] 退出...[/yellow]")
    finally:
        receiver.stop()


__all__ = [
    "VRController",
    "VROffset",
    "VRReceiver",
    "VRData",
    "VRPose",
    "VRButtonEvents",
    "quat_conj",
    "quat_mul",
    "quat_norm",
    "register",
    "get_instance",
    "UDP_IP",
    "UDP_PORT",
    "main",
]
if __name__ == "__main__":
    main()
