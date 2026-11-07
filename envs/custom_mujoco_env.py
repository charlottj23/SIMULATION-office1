from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class CustomRobotEnv(gym.Env):
    """
    커스텀 MuJoCo MJCF(XML) 모델을 불러와 강화학습을 수행하는 Gymnasium 호환 환경.
    Hugging Face Hub 업로드 및 영상 녹화를 위해 `rgb_array` 렌더링과 `human` 뷰어를 모두 지원합니다.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        xml_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        frame_skip: int = 4,
        width: int = 640,
        height: int = 480,
    ):
        super().__init__()

        if xml_path is None:
            xml_path = str(Path(__file__).resolve().parent.parent / "assets" / "custom_robot.xml")
        self.xml_path = xml_path

        # MuJoCo 모델 및 데이터 초기화
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.width = width
        self.height = height

        # 렌더러 및 뷰어 핸들
        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer = None

        # 사이트 ID 캐싱 (End-Effector, Target)
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # Action Space: 2개 모터 토크 제어 [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation Space:
        # [cos(q1), cos(q2), sin(q1), sin(q2), qvel1, qvel2, ee_to_target_dx, ee_to_target_dy] (총 8차원)
        obs_dim = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def _get_obs(self) -> np.ndarray:
        theta = self.data.qpos[:2]
        qvel = self.data.qvel[:2]
        ee_pos = self.data.site_xpos[self._ee_site_id][:2]
        target_pos = self.data.site_xpos[self._target_site_id][:2]
        vec_to_target = ee_pos - target_pos

        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                qvel,
                vec_to_target,
            ]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # 초기 관절 각도 랜덤화
        qpos = self. data.qpos.copy()
        qpos[:2] = self.np_random.uniform(low=-0.2, high=0.2, size=2)

        # 목표 지점(Target) 위치 랜덤 생성 (반경 0.15 ~ 0.40m 이내)
        angle = self.np_random.uniform(low=-np.pi, high=np.pi)
        radius = self.np_random.uniform(low=0.15, high=0.40)
        # 기본 target body 위치(0.25, 0.25)에 대한 slide joint 변위 설정
        qpos[2] = radius * np.cos(angle) - 0.25
        qpos[3] = radius * np.sin(angle) - 0.25

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _get_info(self) -> dict[str, Any]:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        target_pos = self.data.site_xpos[self._target_site_id]
        dist = float(np.linalg.norm(ee_pos[:2] - target_pos[:2]))
        return {
            "distance_to_target": dist,
            "is_success": dist < 0.04,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        # 보상 설계: 목표까지의 거리 페널티 + 제어 에너지 페널티 + 도달 보너스
        reward_dist = -info["distance_to_target"]
        reward_ctrl = -0.05 * float(np.square(action).sum())
        reward_bonus = 1.0 if info["is_success"] else 0.0
        reward = reward_dist + reward_ctrl + reward_bonus

        terminated = False
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

