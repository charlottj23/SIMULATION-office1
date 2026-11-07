from pathlib import Path
import tempfile

import gradio as gr
import gymnasium as gym
import imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

import envs  # Register CustomRobot-v0 and CustomWorld-v0
import run as tidy_run

DEFAULT_WORLD_MODEL = Path("checkpoints/ppo_world_robot.zip")
DEFAULT_2DOF_MODEL = Path("checkpoints/best_model/best_model.zip")


def run_tidy_simulator(
    mode: str,
    can_x: float,
    can_y: float,
    camera_view: str,
    sim_seconds: float,
):
    """
    사무실 정리 모바일 매니퓰레이터(world.xml) 시뮬레이션을 실행하여
    MP4 영상과 수거 상태 요약을 반환합니다.
    """
    xml_path = Path(__file__).resolve().parent / "world.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    can_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
    can_qpos_adr = int(model.jnt_qposadr[can_jnt_id])
    can_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soda_can")
    grasp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    bin_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bin_drop_site")

    mujoco.mj_resetData(model, data)
    data.qpos[can_qpos_adr + 0] = float(can_x)
    data.qpos[can_qpos_adr + 1] = float(can_y)
    data.qpos[can_qpos_adr + 2] = 0.056
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    if "위" in camera_view:
        cam.azimuth = 90
        cam.elevation = -85
        cam.distance = 2.4
        cam.lookat[:] = [0.0, -0.05, 0.20]
    elif "측면" in camera_view:
        cam.azimuth = 180
        cam.elevation = -18
        cam.distance = 1.9
        cam.lookat[:] = [0.0, -0.05, 0.20]
    else:
        cam.azimuth = 135
        cam.elevation = -26
        cam.distance = 2.1
        cam.lookat[:] = [0.0, -0.05, 0.22]

    fps = 50
    dt = model.opt.timestep
    steps_per_frame = max(1, int(round((1.0 / fps) / dt)))
    total_frames = int(float(sim_seconds) * fps)

    frames = []
    max_can_z = 0.056
    min_grasp_dist = 999.0

    if "PPO" in mode and DEFAULT_WORLD_MODEL.exists():
        env = gym.make("CustomWorld-v0", render_mode="rgb_array")
        rl_model = PPO.load(str(DEFAULT_WORLD_MODEL))
        obs, info = env.reset()
        env.unwrapped.data.qpos[can_qpos_adr + 0] = float(can_x)
        env.unwrapped.data.qpos[can_qpos_adr + 1] = float(can_y)
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        obs = env.unwrapped._get_obs()

        for _ in range(min(total_frames, 220)):
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            max_can_z = max(max_can_z, info["can_height"])
            min_grasp_dist = min(min_grasp_dist, info["distance_to_target"])
            renderer.update_scene(env.unwrapped.data, cam)
            frames.append(renderer.render())
            if terminated or truncated:
                break
        final_bin_dist = info["dist_can_to_bin"]
        env.close()
    else:
        import math

        trash_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ["soda_can", "plastic_block", "coffee_cup"]
        ]
        bin_pos = np.array([0.0, -0.52, 0.28])
        cmd = np.array([0.0, 0.0, 0.0, -0.25, 0.45, 0.20, 0.022], dtype=np.float64)
        idx = 0
        phase = 0
        pt = 0.0
        r_stop = 0.415

        for _ in range(total_frames):
            for _ in range(steps_per_frame):
                pt += dt
                obj_pos = data.xpos[trash_ids[idx]].copy()
                if phase == 0:
                    vec = obj_pos[:2]
                    norm = max(0.05, float(np.linalg.norm(vec)))
                    yaw = math.atan2(float(obj_pos[1]), float(obj_pos[0]))
                    bt = obj_pos[:2] - (vec / norm) * r_stop
                    tcmd = [bt[0], bt[1], yaw, 0.35, 0.35, 0.15, 0.022]
                    if pt > 1.3:
                        phase, pt = 1, 0.0
                elif phase == 1:
                    vec = obj_pos[:2] - data.qpos[:2]
                    yaw = math.atan2(float(vec[1]), float(vec[0]))
                    bt = obj_pos[:2] - np.array([math.cos(yaw), math.sin(yaw)]) * r_stop
                    tcmd = [bt[0], bt[1], yaw, 0.78, 0.50, -0.45, 0.022]
                    if pt > 1.1:
                        phase, pt = 2, 0.0
                elif phase == 2:
                    tcmd = [cmd[0], cmd[1], cmd[2], 0.78, 0.50, -0.45, -0.028]
                    if pt > 0.8:
                        phase, pt = 3, 0.0
                elif phase == 3:
                    tcmd = [cmd[0], cmd[1], cmd[2], -0.15, 0.35, 0.15, -0.028]
                    if pt > 1.0:
                        phase, pt = 4, 0.0
                elif phase == 4:
                    yb = math.atan2(float(bin_pos[1]), float(bin_pos[0]))
                    bt = bin_pos[:2] - np.array([math.cos(yb), math.sin(yb)]) * r_stop
                    ytb = math.atan2(float(bin_pos[1] - bt[1]), float(bin_pos[0] - bt[0]))
                    tcmd = [bt[0], bt[1], ytb, 0.35, 0.45, -0.10, -0.028]
                    if pt > 1.5:
                        phase, pt = 5, 0.0
                else:
                    tcmd = [cmd[0], cmd[1], cmd[2], 0.35, 0.45, -0.10, 0.022]
                    if pt > 0.8:
                        idx = (idx + 1) % 3
                        phase, pt = 0, 0.0

                rates = [0.8 * dt, 0.8 * dt, 2.2 * dt, 1.8 * dt, 1.8 * dt, 2.2 * dt, 0.12 * dt]
                for i in range(7):
                    cmd[i] = tidy_run.move_towards(float(cmd[i]), float(tcmd[i]), rates[i])
                data.ctrl[:] = cmd
                mujoco.mj_step(model, data)

                can_z = float(data.xpos[can_body_id, 2])
                max_can_z = max(max_can_z, can_z)
                g_dist = float(np.linalg.norm(data.site_xpos[grasp_site_id] - data.xpos[can_body_id]))
                min_grasp_dist = min(min_grasp_dist, g_dist)

            renderer.update_scene(data, cam)
            frames.append(renderer.render())

        final_bin_dist = float(
            np.linalg.norm(data.xpos[can_body_id, :2] - data.site_xpos[bin_site_id, :2])
        )

    renderer.close()

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.close()
    imageio.mimsave(tmp_video.name, frames, fps=fps)

    summary_md = f"""### 🦾 사무실 정리 로봇 시뮬레이션 결과 (`{mode}`)
- **빈 캔 최고 들어올린 높이**: `{max_can_z:.3f} m`
- **그리퍼-빈 캔 최소 접근 거리**: `{min_grasp_dist:.3f} m`
- **분리수거함까지 최종 거리**: `{final_bin_dist:.3f} m`
"""
    return tmp_video.name, summary_md


with gr.Blocks(title="MuJoCo Office Tidy Mobile Manipulator") as demo:
    gr.Markdown(
        """
        # 🦾 MuJoCo 사무실 정리 모바일 매니퓰레이터 시뮬레이터 (`CustomWorld-v0`)
        주행 베이스(2-DoF) + 4자유도 다관절 로봇 팔 + 평행 그리퍼(집게)로 바닥의 쓰레기를 집어 분리수거함에 넣는 시뮬레이터입니다.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            ctrl_mode = gr.Radio(
                choices=[
                    "Pick & Place 모션 플래너 데모",
                    "PPO 강화학습 정책 (CustomWorld-v0)",
                ],
                value="Pick & Place 모션 플래너 데모",
                label="제어 방식 선택",
            )
            can_x = gr.Slider(0.30, 0.55, value=0.46, step=0.02, label="빈 캔(Soda Can) 초기 X 위치 (m)")
            can_y = gr.Slider(-0.25, 0.25, value=0.22, step=0.02, label="빈 캔(Soda Can) 초기 Y 위치 (m)")
            cam_view = gr.Radio(
                choices=["쿼터뷰 (3D 입체)", "측면뷰 (그리퍼 파지 확인)", "위에서 보기 (평면도)"],
                value="쿼터뷰 (3D 입체)",
                label="카메라 각도",
            )
            sim_sec = gr.Slider(3.0, 10.0, value=6.8, step=0.4, label="시뮬레이션 시간 (초)")
            run_btn = gr.Button("🦾 정리 로봇 시뮬레이션 실행", variant="primary")

        with gr.Column(scale=2):
            video_out = gr.Video(label="MuJoCo 시뮬레이션 영상", autoplay=True)
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_tidy_simulator,
        inputs=[ctrl_mode, can_x, can_y, cam_view, sim_sec],
        outputs=[video_out, metrics_out],
    )


if __name__ == "__main__":
    demo.launch()
