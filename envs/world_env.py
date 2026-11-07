from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class CustomWorldEnv(gym.Env):
    """
    `world.xml` (사무실 정리 모바일 매니퓰레이터 `office_tidy_mobile_manipulator`) 기반의
    Gymnasium 강화학습 환경 (`CustomWorld-v0`).

    에이전트는 7개 액추에이터(모바일 베이스 X/Y, 팔 회전 Yaw, 어깨/팔꿈치/손목 Pitch, 그리퍼 개폐)를
    연속 제어하여 바닥의 빈 캔(`soda_can`)을 집어 올려 초록색 분리수거함(`trash_bin`)에 넣도록 학습합니다.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        xml_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        frame_skip: int = 10,
        width: int = 640,
        height: int = 480,
    ):
        super().__init__()

        if xml_path is None:
            xml_path = str(Path(__file__).resolve().parent.parent / "world.xml")
        self.xml_path = xml_path

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.width = width
        self.height = height

        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer = None

        # 주요 Site 및 Body ID 캐싱
        self._grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center"
        )
        self._bin_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_drop_site"
        )
        self._can_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "soda_can"
        )
        can_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free"
        )
        self._can_qpos_adr = int(self.model.jnt_qposadr[can_jnt_id])

        # 액추에이터 제어 범위
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy().astype(np.float32)
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy().astype(np.float32)

        # Action Space: 정규화된 7차원 연속 행동 [-1.0, 1.0] (내부에서 ctrlrange 변화량으로 매핑)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation Space (23차원):
        # - 로봇 관절 위치(7) & 속도(7)
        # - 그리퍼 중심 좌표(3)
        # - 목표 쓰레기(빈 캔) 좌표(3)
        # - 그리퍼 -> 빈 캔 상대 벡터(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def _get_obs(self) -> np.ndarray:
        robot_qpos = self.data.qpos[:7].astype(np.float32)
        robot_qvel = self.data.qvel[:7].astype(np.float32)
        grasp_pos = self.data.site_xpos[self._grasp_site_id].copy().astype(np.float32)
        can_pos = self.data.xpos[self._can_body_id].copy().astype(np.float32)
        rel_grasp_to_can = can_pos - grasp_pos

        return np.concatenate(
            [robot_qpos, robot_qvel, grasp_pos, can_pos, rel_grasp_to_can]
        )

    def _get_info(self) -> dict[str, Any]:
        grasp_pos = self.data.site_xpos[self._grasp_site_id]
        can_pos = self.data.xpos[self._can_body_id]
        bin_pos = self.data.site_xpos[self._bin_site_id]

        dist_grasp_to_can = float(np.linalg.norm(grasp_pos - can_pos))
        dist_can_to_bin = float(np.linalg.norm(can_pos[:2] - bin_pos[:2]))
        can_lifted = float(can_pos[2]) > 0.12
        in_bin = dist_can_to_bin < 0.14 and float(can_pos[2]) < 0.22

        return {
            "distance_to_target": dist_grasp_to_can,
            "dist_can_to_bin": dist_can_to_bin,
            "can_height": float(can_pos[2]),
            "can_lifted": can_lifted,
            "is_success": in_bin,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # 초기 로봇 팔 준비 자세
        init_ctrl = np.array([0.0, 0.0, 0.0, -0.20, 0.45, 0.15, 0.020], dtype=np.float32)
        self.data.qpos[:6] = init_ctrl[:6]
        self.data.ctrl[:] = init_ctrl

        # 빈 캔 초기 위치를 로봇 전방 반경 내 랜덤 배치
        adr = self._can_qpos_adr
        self.data.qpos[adr + 0] = self.np_random.uniform(0.35, 0.52)
        self.data.qpos[adr + 1] = self.np_random.uniform(-0.20, 0.25)
        self.data.qpos[adr + 2] = 0.056

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 증분 위치 제어 (Delta position control)
        delta_scale = np.array([0.04, 0.04, 0.08, 0.06, 0.06, 0.08, 0.015], dtype=np.float32)
        new_ctrl = np.clip(
            self.data.ctrl + action * delta_scale,
            self._ctrl_low,
            self._ctrl_high,
        )
        self.data.ctrl[:] = new_ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        # 단계별 밀집 보상 (Staged Dense Reward for Pick-and-Place)
        reach_reward = -2.0 * info["distance_to_target"]
        lift_bonus = 2.0 * max(0.0, info["can_height"] - 0.06) if info["distance_to_target"] < 0.10 else 0.0
        place_reward = -1.5 * info["dist_can_to_bin"] if info["can_lifted"] else 0.0
        success_bonus = 10.0 if info["is_success"] else 0.0
        ctrl_penalty = -0.01 * float(np.square(action).sum())

        reward = reach_reward + lift_bonus + place_reward + success_bonus + ctrl_penalty

        terminated = bool(info["is_success"])
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        elif self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
