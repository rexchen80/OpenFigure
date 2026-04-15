import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAME_TO_SDK_ID = {
    "right_shoulder_pitch_joint": 1,
    "right_shoulder_roll_joint": 2,
    "right_shoulder_yaw_joint": 3,
    "right_elbow_joint": 4,
    "right_wrist_roll_joint": 5,
    "right_wrist_yaw_joint": 6,
    "right_wrist_pitch_joint": 7,
    "left_shoulder_pitch_joint": 9,
    "left_shoulder_roll_joint": 10,
    "left_shoulder_yaw_joint": 11,
    "left_elbow_joint": 12,
    "left_wrist_roll_joint": 13,
    "left_wrist_yaw_joint": 14,
    "left_wrist_pitch_joint": 15,
    "waist_yaw_joint": 17,
    "waist_roll_joint": 18,
    "waist_pitch_joint": 19,
    "neck_yaw_joint": 20,
    "neck_roll_joint": 21,
    "neck_pitch_joint": 22,
}
TAKS_SEND_RATE = 50

# 锚点定位资源路径（相对于taks包的上级目录）
_XML = (
    Path(__file__).parent.parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml"
)

JOINT_GROUPS = {
    "left_arm": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
    ],
    "waist": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
    "neck": [
        "neck_yaw_joint",
        "neck_roll_joint",
        "neck_pitch_joint",
    ],
}
END_EFFECTORS = {
    "left_hand": ("left_hand", "left_hand_target"),
    "right_hand": ("right_hand", "right_hand_target"),
}
END_EFFECTOR_FRAME_TYPE = "site"
COLLISION_PAIRS_BODY = [
    (
        ["torso_collision"],
        [
            "right_shoulder_roll_collision",
            "left_shoulder_roll_collision",
            "right_shoulder_yaw_collision",
            "left_shoulder_yaw_collision",
            "right_elbow_collision",
            "left_elbow_collision",
            "right_wrist_roll_collision",
            "left_wrist_roll_collision",
            "right_hand_collision",
            "left_hand_collision",
        ],
    )
]
WAIST_CONFIG = {
    "arm_reach": 0.5,
    "deadzone": 0.15,
    "compensation_gain": 0.7,
    "yaw_smooth": 0.1,
    "pitch_smooth": 0.1,
    "yaw_lateral_deadzone": 0.1,
}
BACKWARD_CONFIG = {
    "threshold": -0.05,
    "blend_range": 0.15,
    "normal_orient_cost": 1.0,
    "backward_orient_cost": 0.1,
    "smooth_alpha": 0.1,
}
ARM_BIAS_CONFIG = {
    "outward_bias": 0.35,
    "downward_bias": 0.15,
    "bias_cost": 0.01,
}
TARGET_FILTER_CONFIG = {
    "max_linear_speed": 1.8,
    "max_angular_speed": 8.0,
    "large_step_distance": 0.18,
    "large_step_blend": 0.25,
    "high_reach_height": 0.12,
    "high_reach_orient_cost": 0.35,
    "high_reach_damping_scale": 0.55,
}
_NECK_MAX_YAW_STEP = 0.05
_NECK_MAX_PITCH_STEP = 0.05
_UNIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_ZERO_VEC3 = np.zeros(3, dtype=np.float64)
DEFAULT_LEFT_HAND_POS = np.array([0.401749, 0.17502062, 0.76653706], dtype=np.float64)
DEFAULT_RIGHT_HAND_POS = np.array([0.401749, -0.17502062, 0.76653706], dtype=np.float64)
DEFAULT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _compute_waist_yaw(hcx, hcy, wx, wy, inner_r, outer_r, lat_dz):
    dx, dy = hcx - wx, hcy - wy
    abs_dy = abs(dy)
    if abs_dy <= lat_dz:
        return 0.0
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= inner_r:
        return 0.0
    angle = math.atan2(dy, dx)
    lat_blend = min((abs_dy - lat_dz) / 0.15, 1.0)
    if dist >= outer_r:
        return angle * lat_blend
    dist_blend = (dist - inner_r) / (outer_r - inner_r)
    return angle * dist_blend * lat_blend


def _compute_waist_pitch(forward_dist, arm_reach, deadzone, gain, hand_h, waist_h):
    fwd_threshold = arm_reach - deadzone
    forward_pitch = 0.0
    if forward_dist > fwd_threshold:
        forward_pitch = min((forward_dist - fwd_threshold) * gain, 0.45)
    backward_pitch = 0.0
    backward_origin = 0.0
    backward_blend_range = 0.1
    if forward_dist < backward_origin:
        backward_depth = -forward_dist - -backward_origin
        backward_pitch = -min(backward_depth * gain * 1.2, 0.4)
    elif forward_dist < backward_origin + backward_blend_range:
        blend = (
            backward_origin + backward_blend_range - forward_dist
        ) / backward_blend_range
        backward_depth = max(0.0, -forward_dist - -backward_origin)
        backward_pitch = -min(backward_depth * gain * 1.2, 0.4) * blend
    height_pitch = 0.0
    hand_height_diff = hand_h - waist_h
    height_threshold = -0.05
    if hand_height_diff < height_threshold:
        height_pitch = min(
            abs(hand_height_diff - height_threshold) * gain * 1.8,
            0.55,
        )
    if backward_pitch < 0 and height_pitch > 0:
        blend_factor = min(height_pitch / 0.3, 1.0)
        return (
            backward_pitch * (1.0 - blend_factor * 0.7)
            + height_pitch * blend_factor * 0.5
        )
    if backward_pitch < 0:
        return backward_pitch
    return max(forward_pitch, height_pitch)


def _compute_local_fwd(hx, hy, wx, wy, yaw):
    dx, dy = hx - wx, hy - wy
    cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)
    return dx * cos_yaw - dy * sin_yaw


def _compute_neck_target(dx, dy, dz, dist, waist_yaw, inner_r, outer_r):
    if dist <= inner_r:
        return 0.0, 0.0
    inv_dist = 1.0 / dist
    ndx, ndy, ndz = dx * inv_dist, dy * inv_dist, dz * inv_dist
    horiz_angle = math.atan2(ndy, ndx)
    rel_yaw = _wrap_angle(horiz_angle - waist_yaw)
    horiz_dist = math.sqrt(ndx * ndx + ndy * ndy)
    pitch_angle = -math.atan2(ndz, horiz_dist)
    clamped_yaw = max(-1.2, min(1.2, rel_yaw))
    clamped_pitch = max(-0.5, min(0.8, pitch_angle))
    if dist >= outer_r:
        return clamped_yaw, clamped_pitch
    blend = (dist - inner_r) / (outer_r - inner_r)
    return clamped_yaw * blend, clamped_pitch * blend


def _slerp(q0, q1, t):
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        lerp = q0 + t * (q1 - q0)
        norm = math.sqrt(
            lerp[0] * lerp[0]
            + lerp[1] * lerp[1]
            + lerp[2] * lerp[2]
            + lerp[3] * lerp[3]
        )
        return lerp / norm
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    return (
        math.sin((1.0 - t) * angle) / sin_angle * q0
        + math.sin(t * angle) / sin_angle * q1
    )


def _quat_angle(q0, q1):
    dot = abs(float(np.dot(q0, q1)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def _ensure_quat_continuity(q_new, q_prev):
    return -q_new if np.dot(q_new, q_prev) < 0.0 else q_new


def quat_from_euler(roll, pitch, yaw):
    rot = Rotation.from_euler("xyz", [roll, pitch, yaw])
    xyzw = rot.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def euler_from_quat(quat):
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return rot.as_euler("xyz")


def _compute_neck_angles(
    hands_center,
    ref_pos,
    prev_yaw,
    prev_pitch,
    waist_yaw,
    max_yaw_step=_NECK_MAX_YAW_STEP,
    max_pitch_step=_NECK_MAX_PITCH_STEP,
):
    delta = hands_center - ref_pos
    dist = math.sqrt(float(delta[0]) ** 2 + float(delta[1]) ** 2 + float(delta[2]) ** 2)
    if dist < 1e-06:
        return prev_yaw, prev_pitch
    target_yaw, target_pitch = _compute_neck_target(
        float(delta[0]),
        float(delta[1]),
        float(delta[2]),
        dist,
        waist_yaw,
        0.3,
        0.6,
    )
    yaw_step = max(
        -max_yaw_step,
        min(max_yaw_step, _wrap_angle(target_yaw - prev_yaw) * 0.1),
    )
    pitch_step = max(
        -max_pitch_step,
        min(max_pitch_step, (target_pitch - prev_pitch) * 0.1),
    )
    return prev_yaw + yaw_step, prev_pitch + pitch_step


@dataclass
class ResetState:
    active: bool = False
    alpha: float = 0.0
    start_q: Optional[np.ndarray] = None
    start_pos: Dict[str, np.ndarray] = field(default_factory=dict)
    start_quat: Dict[str, np.ndarray] = field(default_factory=dict)
    start_waist_yaw: float = 0.0


@dataclass
class ProtectionState:
    frozen: bool = False
    frozen_pos: Dict[str, np.ndarray] = field(default_factory=dict)
    frozen_quat: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class MITParams:
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


@dataclass
class IKResult:
    joint_positions: Dict[int, float]
    joint_torques: Dict[int, float]
    mit_params: Dict[int, MITParams]
    left_hand_pos: np.ndarray
    right_hand_pos: np.ndarray


class EndEffectorProxy:
    def __init__(self, solver, name):
        self._solver = solver
        self._name = name

    def eef(self):
        return self._solver.get_end_effector_position(self._name)

    def pos(self):
        position, _quat = self._solver.get_end_effector_position(self._name)
        return position

    def quat(self):
        _pos, quaternion = self._solver.get_end_effector_position(self._name)
        return quaternion

    def euler(self):
        _pos, quaternion = self._solver.get_end_effector_position(self._name)
        return euler_from_quat(quaternion)

    def mocap_pos(self):
        mocap_id = self._solver.mocap_ids.get(self._name, -1)
        if mocap_id < 0:
            return self.pos()
        return self._solver.data.mocap_pos[mocap_id].copy()

    def mocap_quat(self):
        mocap_id = self._solver.mocap_ids.get(self._name, -1)
        if mocap_id < 0:
            return self.quat()
        return self._solver.data.mocap_quat[mocap_id].copy()

    def home_pos(self):
        return self._solver.init_pos.get(self._name, _ZERO_VEC3.copy())

    def home_quat(self):
        return self._solver.init_quat.get(self._name, _UNIT_QUAT.copy())

    def home_euler(self):
        home_q = self._solver.init_quat.get(self._name, _UNIT_QUAT.copy())
        return euler_from_quat(home_q)

    def set_target_quat(self, position, quaternion=None):
        if quaternion is None:
            quaternion = self.home_quat()
        self._solver.set_end_effector_target(
            self._name,
            np.asarray(position, dtype=np.float64),
            np.asarray(quaternion, dtype=np.float64),
        )

    def set_target_euler(self, position, roll=None, pitch=None, yaw=None):
        if roll is None or pitch is None or yaw is None:
            home_rpy = self.home_euler()
            roll = roll if roll is not None else home_rpy[0]
            pitch = pitch if pitch is not None else home_rpy[1]
            yaw = yaw if yaw is not None else home_rpy[2]
        target_quat = quat_from_euler(roll, pitch, yaw)
        self._solver.set_end_effector_target(
            self._name,
            np.asarray(position, dtype=np.float64),
            target_quat,
        )


class HalfBodyIKSolver:
    def __init__(self, xml_path=None):
        self.xml_path = xml_path or _XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path.as_posix())
        self.cfg = mink.Configuration(self.model)
        self.model, self.data = self.cfg.model, self.cfg.data
        self._init_indices()
        self._init_tasks()
        self._init_state()
        self.dt = 1.0 / TAKS_SEND_RATE

    def _init_indices(self):
        self.joint_idx = {
            grp: [self.model.jnt_dofadr[self.model.joint(jname).id] for jname in joints]
            for (grp, joints) in JOINT_GROUPS.items()
        }
        self.neck_dof = [int(idx) for idx in self.joint_idx["neck"]]
        self.waist_dof = [int(idx) for idx in self.joint_idx["waist"]]
        self.waist_yaw_qpos = self.model.jnt_qposadr[
            self.model.joint("waist_yaw_joint").id
        ]
        self.waist_roll_qpos = self.model.jnt_qposadr[
            self.model.joint("waist_roll_joint").id
        ]
        self.waist_pitch_qpos = self.model.jnt_qposadr[
            self.model.joint("waist_pitch_joint").id
        ]
        self.neck_yaw_qpos = self.model.jnt_qposadr[
            self.model.joint("neck_yaw_joint").id
        ]
        self.neck_pitch_qpos = self.model.jnt_qposadr[
            self.model.joint("neck_pitch_joint").id
        ]
        self.neck_roll_qpos = self.model.jnt_qposadr[
            self.model.joint("neck_roll_joint").id
        ]
        self._sdk_ids = []
        self._qpos_indices = []
        self._dof_indices = []
        for _grp, joints in JOINT_GROUPS.items():
            for jname in joints:
                sdk_id = JOINT_NAME_TO_SDK_ID.get(jname)
                if sdk_id:
                    qpos_idx = self.model.jnt_qposadr[self.model.joint(jname).id]
                    dof_idx = self.model.jnt_dofadr[self.model.joint(jname).id]
                    self._sdk_ids.append(sdk_id)
                    self._qpos_indices.append(qpos_idx)
                    self._dof_indices.append(dof_idx)
        self._qpos_indices_arr = np.array(self._qpos_indices, dtype=np.intp)
        self._dof_indices_arr = np.array(self._dof_indices, dtype=np.intp)
        self._freeze_dof = self.neck_dof + self.waist_dof
        self._result_dict = dict.fromkeys(self._sdk_ids, 0.0)
        self._torque_dict = dict.fromkeys(self._sdk_ids, 0.0)
        self._pos_buf = np.zeros(3, dtype=np.float64)
        self._torso_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "torso_collision",
        )
        arm_geom_names = [
            "right_wrist_roll_collision",
            "left_wrist_roll_collision",
            "right_hand_collision",
            "left_hand_collision",
        ]
        self._arm_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            for gname in arm_geom_names
        ]
        self._arm_geom_names_cached = arm_geom_names
        self._site_ids = {}
        for eef_name, (
            site_name,
            _target,
        ) in END_EFFECTORS.items():
            self._site_ids[eef_name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
        self._eef_pos_buf = np.zeros(3, dtype=np.float64)
        self._eef_quat_buf = np.empty(4, dtype=np.float64)

    def _init_tasks(self):
        self.tasks = [
            mink.FrameTask(
                "base_link",
                "body",
                position_cost=1e6,
                orientation_cost=1e6,
            ),
            mink.PostureTask(self.model, cost=0.001),
        ]
        self.ee_tasks = {}
        for eef_name, (
            site_name,
            _target,
        ) in END_EFFECTORS.items():
            ee_task = mink.FrameTask(
                site_name,
                END_EFFECTOR_FRAME_TYPE,
                position_cost=2e1,
                orientation_cost=2.0,
            )
            self.tasks.append(ee_task)
            self.ee_tasks[eef_name] = ee_task
        damping_weights = np.ones(self.model.nv, dtype=np.float64) * 1.0
        self._base_damping_weights = damping_weights.copy()
        for arm in ["left_arm", "right_arm"]:
            for jname in JOINT_GROUPS[arm]:
                dof_idx = self.model.jnt_dofadr[self.model.joint(jname).id]
                if "shoulder" in jname or "elbow" in jname:
                    damping_weights[dof_idx] = 3.0
                elif "wrist" in jname:
                    damping_weights[dof_idx] = 1.5
        self._base_damping_weights[:] = damping_weights
        self.damping_task = mink.DampingTask(self.model, cost=damping_weights)
        self.tasks.append(self.damping_task)
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.CollisionAvoidanceLimit(
                self.model,
                COLLISION_PAIRS_BODY,
                gain=2.0,
                minimum_distance_from_collisions=0.001,
                collision_detection_distance=0.1,
            ),
        ]
        self.cfg.update_from_keyframe("home")
        mujoco.mj_forward(self.model, self.data)
        self.tasks[0].set_target_from_configuration(self.cfg)
        self.tasks[1].set_target_from_configuration(self.cfg)
        for eef_name, (
            site_name,
            target_name,
        ) in END_EFFECTORS.items():
            mink.move_mocap_to_frame(
                self.model,
                self.data,
                target_name,
                site_name,
                END_EFFECTOR_FRAME_TYPE,
            )
            self.ee_tasks[eef_name].set_target_from_configuration(self.cfg)
        mujoco.mj_forward(self.model, self.data)

    def _init_state(self):
        self.init_q = self.cfg.q.copy()
        self.mocap_ids = {}
        self.init_pos = {}
        self.init_quat = {}
        self.prev_quat = {}
        self.prev_target_pos = {}
        self.prev_target_quat = {}
        for eef_name, (
            site_name,
            target_body,
        ) in END_EFFECTORS.items():
            mocap_id = self.model.body(target_body).mocapid[0]
            self.mocap_ids[eef_name] = mocap_id
            site_id = self._site_ids[eef_name]
            self.init_pos[eef_name] = self.data.site_xpos[site_id].copy()
            _quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(_quat, self.data.site_xmat[site_id])
            self.init_quat[eef_name] = _quat
            self.prev_quat[eef_name] = _quat.copy()
            self.prev_target_pos[eef_name] = self.init_pos[eef_name].copy()
            self.prev_target_quat[eef_name] = _quat.copy()
            print(
                f"[IK Init] {eef_name} initial pos: {self.init_pos[eef_name]}, quat: {self.init_quat[eef_name]}"
            )
        self.waist_init_pos = self.data.xpos[self.model.body("base_link").id].copy()
        self._neck_ref_pos = self.data.xpos[self.model.body("neck_yaw_link").id].copy()
        self.prev_neck_yaw = 0.0
        self.prev_neck_pitch = 0.0
        self.prev_waist_yaw = 0.0
        self.prev_waist_pitch = 0.0
        self.smooth_left_backward = 0.0
        self.smooth_right_backward = 0.0
        self.reset_state = ResetState()
        self.protection_state = ProtectionState()
        self.first_solve = True

    def _check_arm_in_torso(self):
        torso_id = self._torso_geom_id
        if torso_id < 0:
            return False
        torso_pos = self.data.geom_xpos[torso_id]
        all_geom_pos = self.data.geom_xpos
        for arm_idx, arm_geom_id in enumerate(self._arm_geom_ids):
            if arm_geom_id < 0:
                continue
            rel = all_geom_pos[arm_geom_id] - torso_pos
            dist_xy_sq = rel[0] * rel[0] + rel[1] * rel[1]
            if dist_xy_sq < 0.0144 and abs(rel[2]) < 0.2:
                print(
                    f"[软保护] {self._arm_geom_names_cached[arm_idx]} 穿入躯干: dist_xy={math.sqrt(dist_xy_sq):.3f} rel_z={rel[2]:.3f}"
                )
                return True
        return False

    def set_end_effector_target(self, name, position, quaternion):
        if name not in self.mocap_ids:
            return
        mocap_id = self.mocap_ids[name]
        if self.protection_state.frozen:
            self.data.mocap_pos[mocap_id][:] = self.protection_state.frozen_pos[name]
            self.data.mocap_quat[mocap_id][:] = self.protection_state.frozen_quat[name]
            return
        quaternion = _ensure_quat_continuity(quaternion, self.prev_quat[name])
        prev_pos = self.prev_target_pos[name]
        prev_quat = self.prev_target_quat[name]
        filter_cfg = TARGET_FILTER_CONFIG
        max_step = filter_cfg["max_linear_speed"] * self.dt
        delta = position - prev_pos
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > filter_cfg["large_step_distance"]:
            blend = filter_cfg["large_step_blend"]
            position = prev_pos + delta * blend
            delta = position - prev_pos
            delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_step and delta_norm > 1e-9:
            position = prev_pos + delta * (max_step / delta_norm)
        quat_angle = _quat_angle(prev_quat, quaternion)
        max_quat_angle = filter_cfg["max_angular_speed"] * self.dt
        if quat_angle > max_quat_angle:
            quaternion = _slerp(prev_quat, quaternion, max_quat_angle / quat_angle)
        self.prev_quat[name][:] = quaternion
        self.prev_target_pos[name][:] = position
        self.prev_target_quat[name][:] = quaternion
        self.data.mocap_pos[mocap_id][:] = position
        self.data.mocap_quat[mocap_id][:] = quaternion

    def _freeze_protection(self):
        if self.protection_state.frozen:
            return
        self.protection_state.frozen = True
        for eef_name, mocap_id in self.mocap_ids.items():
            self.protection_state.frozen_pos[eef_name] = self.data.mocap_pos[
                mocap_id
            ].copy()
            self.protection_state.frozen_quat[eef_name] = self.data.mocap_quat[
                mocap_id
            ].copy()
        print("[软保护] mocap进入躯干区域，已冻结。双击B键或键盘退格键复位。")

    def unfreeze_and_reset(self):
        if self.protection_state.frozen:
            self.protection_state.frozen = False
            self.reset()
            print("[软保护] 已解冻，开始复位...")

    def get_end_effector_position(self, name):
        site_id = self._site_ids.get(name, -1)
        if site_id < 0:
            return _ZERO_VEC3.copy(), _UNIT_QUAT.copy()
        pos = self.data.site_xpos[site_id].copy()
        mujoco.mju_mat2Quat(self._eef_quat_buf, self.data.site_xmat[site_id])
        return pos, self._eef_quat_buf.copy()

    def _compute_backward_factor(self, forward_dist):
        threshold, blend_range = (
            BACKWARD_CONFIG["threshold"],
            BACKWARD_CONFIG["blend_range"],
        )
        if forward_dist >= threshold:
            return 0.0
        if forward_dist <= threshold - blend_range:
            return 1.0
        return (threshold - forward_dist) / blend_range

    def solve(self, ramp_progress=1.0):
        if self.protection_state.frozen:
            return self._result_dict
        left_mocap_id = self.mocap_ids["left_hand"]
        right_mocap_id = self.mocap_ids["right_hand"]
        left_pos = self.data.mocap_pos[left_mocap_id]
        right_pos = self.data.mocap_pos[right_mocap_id]
        hands_center = (left_pos + right_pos) * 0.5
        waist_ref = self.waist_init_pos
        left_dx, left_dy = (
            left_pos[0] - waist_ref[0],
            left_pos[1] - waist_ref[1],
        )
        right_dx, right_dy = (
            right_pos[0] - waist_ref[0],
            right_pos[1] - waist_ref[1],
        )
        left_dist = math.sqrt(left_dx * left_dx + left_dy * left_dy)
        right_dist = math.sqrt(right_dx * right_dx + right_dy * right_dy)
        max_dist = max(left_dist, right_dist)
        reach_inner, reach_outer = 0.2, 0.4
        reach_blend = (
            0.0
            if max_dist <= reach_inner
            else 1.0
            if max_dist >= reach_outer
            else (max_dist - reach_inner) / (reach_outer - reach_inner)
        )
        lat_dz = WAIST_CONFIG["yaw_lateral_deadzone"]
        target_waist_yaw = _compute_waist_yaw(
            float(hands_center[0]),
            float(hands_center[1]),
            float(waist_ref[0]),
            float(waist_ref[1]),
            reach_inner,
            reach_outer,
            lat_dz,
        )
        max_hand_h = max(float(left_pos[2]), float(right_pos[2]))
        init_max_hand_h = max(
            float(self.init_pos["left_hand"][2]),
            float(self.init_pos["right_hand"][2]),
        )
        hand_height_delta = max_hand_h - init_max_hand_h
        yaw_smooth = WAIST_CONFIG["yaw_smooth"]
        if hand_height_delta > 0.05:
            yaw_smooth *= (
                0.1
                if hand_height_delta >= 0.3
                else 1.0 - 0.9 * (hand_height_delta - 0.05) / 0.25
            )
        yaw_diff = _wrap_angle(target_waist_yaw - self.prev_waist_yaw)
        waist_yaw = self.prev_waist_yaw + yaw_diff * yaw_smooth
        backward_cfg = BACKWARD_CONFIG
        left_fwd = _compute_local_fwd(
            float(left_pos[0]),
            float(left_pos[1]),
            float(waist_ref[0]),
            float(waist_ref[1]),
            waist_yaw,
        )
        right_fwd = _compute_local_fwd(
            float(right_pos[0]),
            float(right_pos[1]),
            float(waist_ref[0]),
            float(waist_ref[1]),
            waist_yaw,
        )
        smooth_alpha = backward_cfg["smooth_alpha"]
        self.smooth_left_backward = (
            smooth_alpha * self._compute_backward_factor(left_fwd)
            + (1.0 - smooth_alpha) * self.smooth_left_backward
        )
        self.smooth_right_backward = (
            smooth_alpha * self._compute_backward_factor(right_fwd)
            + (1.0 - smooth_alpha) * self.smooth_right_backward
        )
        normal_cost, backward_cost = (
            backward_cfg["normal_orient_cost"],
            backward_cfg["backward_orient_cost"],
        )
        left_height_delta = float(left_pos[2]) - float(self.init_pos["left_hand"][2])
        right_height_delta = float(right_pos[2]) - float(self.init_pos["right_hand"][2])
        high_reach_h = TARGET_FILTER_CONFIG["high_reach_height"]
        left_high_reach = min(max(left_height_delta / high_reach_h, 0.0), 1.0)
        right_high_reach = min(max(right_height_delta / high_reach_h, 0.0), 1.0)
        left_orient_cost = (
            normal_cost * (1.0 - self.smooth_left_backward)
            + backward_cost * self.smooth_left_backward
        )
        right_orient_cost = (
            normal_cost * (1.0 - self.smooth_right_backward)
            + backward_cost * self.smooth_right_backward
        )
        min_orient_cost = TARGET_FILTER_CONFIG["high_reach_orient_cost"]
        left_orient_cost = (
            left_orient_cost * (1.0 - left_high_reach)
            + min_orient_cost * left_high_reach
        )
        right_orient_cost = (
            right_orient_cost * (1.0 - right_high_reach)
            + min_orient_cost * right_high_reach
        )
        self.ee_tasks["left_hand"].set_orientation_cost(left_orient_cost)
        self.ee_tasks["right_hand"].set_orientation_cost(right_orient_cost)
        damping_scale = 1.0 - max(left_high_reach, right_high_reach) * (
            1.0 - TARGET_FILTER_CONFIG["high_reach_damping_scale"]
        )
        self.damping_task.set_cost(self._base_damping_weights * damping_scale)
        if self.smooth_left_backward > 0.75 or self.smooth_right_backward > 0.75:
            waist_yaw = self.prev_waist_yaw
        self.prev_waist_yaw = waist_yaw
        left_fwd2 = _compute_local_fwd(
            float(left_pos[0]),
            float(left_pos[1]),
            float(waist_ref[0]),
            float(waist_ref[1]),
            waist_yaw,
        )
        right_fwd2 = _compute_local_fwd(
            float(right_pos[0]),
            float(right_pos[1]),
            float(waist_ref[0]),
            float(waist_ref[1]),
            waist_yaw,
        )
        dominant_fwd = left_fwd2 if abs(left_fwd2) > abs(right_fwd2) else right_fwd2
        min_hand_h = min(float(left_pos[2]), float(right_pos[2]))
        init_min_hand_h = min(
            float(self.init_pos["left_hand"][2]),
            float(self.init_pos["right_hand"][2]),
        )
        low_hand_delta = min_hand_h - init_min_hand_h
        low_blend = (
            0.0
            if low_hand_delta >= -0.02
            else 1.0
            if low_hand_delta <= -0.1
            else (-0.02 - low_hand_delta) / 0.08
        )
        waist_cfg = WAIST_CONFIG
        raw_pitch = _compute_waist_pitch(
            dominant_fwd,
            waist_cfg["arm_reach"],
            waist_cfg["deadzone"],
            waist_cfg["compensation_gain"],
            min_hand_h,
            init_min_hand_h,
        )
        target_pitch = raw_pitch * max(reach_blend, low_blend)
        waist_pitch = (
            self.prev_waist_pitch
            + (target_pitch - self.prev_waist_pitch) * waist_cfg["pitch_smooth"]
        )
        self.prev_waist_pitch = waist_pitch
        if self.reset_state.active:
            self.reset_state.alpha += self.dt * 0.5
            reset_alpha = min(1.0, self.reset_state.alpha)
            reset_beta = 1.0 - reset_alpha
            for eef_name, mocap_id in self.mocap_ids.items():
                self.data.mocap_pos[mocap_id][:] = (
                    reset_beta * self.reset_state.start_pos[eef_name]
                    + reset_alpha * self.init_pos[eef_name]
                )
                self.data.mocap_quat[mocap_id][:] = _slerp(
                    self.reset_state.start_quat[eef_name],
                    self.init_quat[eef_name],
                    reset_alpha,
                )
                self.prev_quat[eef_name][:] = self.data.mocap_quat[mocap_id]
                self.prev_target_pos[eef_name][:] = self.data.mocap_pos[mocap_id]
                self.prev_target_quat[eef_name][:] = self.data.mocap_quat[mocap_id]
            hands_center = (
                self.data.mocap_pos[left_mocap_id] + self.data.mocap_pos[right_mocap_id]
            ) * 0.5
            waist_yaw = self.reset_state.start_waist_yaw * reset_beta
            waist_pitch = self.prev_waist_pitch * reset_beta
            if reset_alpha >= 1.0:
                self.reset_state.active = False
                self.prev_waist_yaw = 0.0
                self.prev_waist_pitch = 0.0
                self.smooth_left_backward = 0.0
                self.smooth_right_backward = 0.0
        if self.first_solve:
            neck_yaw, neck_pitch = 0.0, 0.0
            self.prev_neck_yaw = 0.0
            self.prev_neck_pitch = 0.0
            self.first_solve = False
        else:
            neck_yaw, neck_pitch = _compute_neck_angles(
                hands_center,
                self._neck_ref_pos,
                self.prev_neck_yaw,
                self.prev_neck_pitch,
                waist_yaw,
            )
        self.prev_neck_yaw = neck_yaw
        self.prev_neck_pitch = neck_pitch
        for eef_name, mocap_id in self.mocap_ids.items():
            self.ee_tasks[eef_name].set_target(
                mink.SE3.from_mocap_id(self.data, mocap_id)
            )
        qpos = self.cfg.q
        qpos[self.waist_yaw_qpos] = waist_yaw
        qpos[self.waist_pitch_qpos] = waist_pitch
        qpos[self.waist_roll_qpos] = 0.0
        self.cfg.update(qpos)
        mujoco.mj_forward(self.model, self.data)
        if not hasattr(self, "_freeze_constraint"):
            self._freeze_constraint = mink.DofFreezingTask(
                self.model, dof_indices=self._freeze_dof
            )
        constraints = [self._freeze_constraint]
        try:
            vel = mink.solve_ik(
                self.cfg,
                self.tasks,
                self.dt,
                "daqp",
                limits=self.limits,
                constraints=constraints,
            )
        except mink.exceptions.NoSolutionFound:
            vel = mink.solve_ik(
                self.cfg,
                self.tasks,
                self.dt,
                "daqp",
                limits=self.limits,
                constraints=[],
            )
        self.cfg.integrate_inplace(vel, self.dt)
        qpos = self.cfg.q
        qpos[self.waist_yaw_qpos] = waist_yaw
        qpos[self.waist_pitch_qpos] = waist_pitch
        qpos[self.waist_roll_qpos] = 0.0
        qpos[self.neck_yaw_qpos] = neck_yaw * ramp_progress
        qpos[self.neck_pitch_qpos] = neck_pitch * ramp_progress
        qpos[self.neck_roll_qpos] = 0.0
        self.cfg.update(qpos)
        self.data.qfrc_applied[:] = self.data.qfrc_bias
        if not self.reset_state.active and not self.protection_state.frozen:
            if self._check_arm_in_torso():
                self._freeze_protection()
        solved_qpos = qpos[self._qpos_indices_arr]
        for idx, sdk_id in enumerate(self._sdk_ids):
            self._result_dict[sdk_id] = float(solved_qpos[idx])
        return self._result_dict

    def get_joint_torques(self):
        torques = self.data.qfrc_bias[self._dof_indices_arr]
        for idx, sdk_id in enumerate(self._sdk_ids):
            self._torque_dict[sdk_id] = float(torques[idx])
        return self._torque_dict

    def reset(self):
        self.reset_state.active = True
        self.reset_state.alpha = 0.0
        self.reset_state.start_q = self.cfg.q.copy()
        for eef_name, mocap_id in self.mocap_ids.items():
            self.reset_state.start_pos[eef_name] = self.data.mocap_pos[mocap_id].copy()
            self.reset_state.start_quat[eef_name] = self.data.mocap_quat[
                mocap_id
            ].copy()
            self.prev_target_pos[eef_name][:] = self.data.mocap_pos[mocap_id]
            self.prev_target_quat[eef_name][:] = self.data.mocap_quat[mocap_id]
        self.reset_state.start_waist_yaw = self.prev_waist_yaw

    @property
    def is_resetting(self):
        return self.reset_state.active


class IKController:
    def __init__(self, device_type="Semi-Taks-T1", feedforward_scale=1.0):
        self.device_type = device_type
        self.ik = HalfBodyIKSolver()
        self.feedforward_scale = feedforward_scale
        self._kp_dict = {}
        self._kd_dict = {}
        self._mit_cmd_cache = {
            sdk_id: {
                "q": 0.0,
                "dq": 0.0,
                "tau": 0.0,
                "kp": 0.0,
                "kd": 0.0,
            }
            for sdk_id in self.ik._sdk_ids
        }
        self.left_hand = EndEffectorProxy(self.ik, "left_hand")
        self.right_hand = EndEffectorProxy(self.ik, "right_hand")

    @property
    def dt(self):
        return self.ik.dt

    @dt.setter
    def dt(self, value):
        self.ik.dt = value

    @property
    def model(self):
        return self.ik.model

    @property
    def data(self):
        return self.ik.data

    def step(
        self,
        left_pos=None,
        left_quat=None,
        right_pos=None,
        right_quat=None,
        ramp_progress=1.0,
    ):
        if not self.ik.is_resetting:
            if left_pos is not None:
                self.left_hand.set_target_quat(left_pos, left_quat)
            if right_pos is not None:
                self.right_hand.set_target_quat(right_pos, right_quat)
        return self.solve(ramp_progress)

    def solve(self, ramp_progress=1.0):
        joint_pos = self.ik.solve(ramp_progress)
        joint_tau = self.ik.get_joint_torques()
        ff_scale = self.feedforward_scale
        kp_dict = self._kp_dict
        kd_dict = self._kd_dict
        mit_cache = self._mit_cmd_cache
        for sdk_id, qval in joint_pos.items():
            cmd = mit_cache[sdk_id]
            cmd["q"] = float(qval)
            cmd["tau"] = joint_tau.get(sdk_id, 0.0) * ff_scale
            cmd["kp"] = kp_dict.get(sdk_id, 0.0)
            cmd["kd"] = kd_dict.get(sdk_id, 0.0)
        return mit_cache

    @property
    def joint_positions(self):
        return self.ik._result_dict

    @property
    def joint_torques(self):
        return self.ik._torque_dict

    def reset(self):
        self.ik.reset()

    @property
    def is_resetting(self):
        return self.ik.is_resetting

    @property
    def is_frozen(self):
        return self.ik.protection_state.frozen

    @property
    def protection_state(self):
        return self.ik.protection_state

    def unfreeze(self):
        self.ik.unfreeze_and_reset()


_instance = None


def register(device_type="Semi-Taks-T1", feedforward_scale=1.0):
    global _instance
    _instance = IKController(
        device_type=device_type,
        feedforward_scale=feedforward_scale,
    )
    return _instance


def get_instance():
    return _instance


__all__ = [
    "IKController",
    "IKResult",
    "MITParams",
    "EndEffectorProxy",
    "HalfBodyIKSolver",
    "JOINT_NAME_TO_SDK_ID",
    "TAKS_SEND_RATE",
    "DEFAULT_LEFT_HAND_POS",
    "DEFAULT_RIGHT_HAND_POS",
    "DEFAULT_QUAT",
    "register",
    "get_instance",
    "quat_from_euler",
    "euler_from_quat",
]
