import argparse
from pathlib import Path
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

import envs  # Register CustomRobot-v0


def train(
    env_id: str = "CustomWorld-v0",
    total_timesteps: int = 50_000,
    n_envs: int = 4,
    record_video: bool = True,
):
    log_dir = Path("logs")
    model_dir = Path("checkpoints")
    video_dir = Path("videos")
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    video_dir.mkdir(exist_ok=True)

    model_tag = "ppo_world_robot" if env_id == "CustomWorld-v0" else "ppo_custom_robot"

    print(f"[1/3] {env_id} 병렬 학습 환경({n_envs}개) 생성 중...")
    vec_env = make_vec_env(env_id, n_envs=n_envs, seed=42)
    eval_env = make_vec_env(env_id, n_envs=1, seed=123)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / f"best_{model_tag}"),
        log_path=str(log_dir),
        eval_freq=max(2500 // n_envs, 500),
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
    )

    print(f"[2/3] PPO 강화학습 시작 (총 {total_timesteps:,} 스텝)...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    final_model_path = model_dir / model_tag
    model.save(str(final_model_path))
    print(f"[완료] 모델 저장됨: {final_model_path}.zip")

    if record_video:
        print("[3/3] 학습된 에이전트 리플레이 영상(.mp4) 생성 중...")
        rec_env = make_vec_env(
            env_id,
            n_envs=1,
            env_kwargs={"render_mode": "rgb_array"},
        )
        rec_env = VecVideoRecorder(
            rec_env,
            str(video_dir),
            record_video_trigger=lambda x: x == 0,
            video_length=200,
            name_prefix=f"{model_tag}-replay",
        )
        obs = rec_env.reset()
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = rec_env.step(action)
        rec_env.close()
        print(f"[완료] 리플레이 영상이 '{video_dir}/' 폴더에 저장되었습니다.")

    vec_env.close()
    eval_env.close()


def enjoy(
    env_id: str = "CustomWorld-v0",
    model_path: str = "checkpoints/ppo_world_robot.zip",
    episodes: int = 3,
):
    """학습된 모델을 MuJoCo 3D 인터랙티브 뷰어로 실행합니다."""
    env = gym.make(env_id, render_mode="human")
    model = PPO.load(model_path, env=env)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            time.sleep(env.unwrapped.dt)
        print(f"Episode {ep + 1}: Reward = {ep_reward:.2f}, Final Dist = {info['distance_to_target']:.4f}m")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo 강화학습 시뮬레이터 학습 및 뷰어 실행")
    parser.add_argument("--env-id", choices=["CustomWorld-v0", "CustomRobot-v0"], default="CustomWorld-v0", help="학습 환경 ID")
    parser.add_argument("--mode", choices=["train", "enjoy"], default="train", help="실행 모드: train 또는 enjoy")
    parser.add_argument("--timesteps", type=int, default=50000, help="총 학습 타임스텝 수")
    parser.add_argument("--model-path", type=str, default="checkpoints/ppo_world_robot.zip", help="불러올 모델 경로")
    args = parser.parse_args()

    if args.mode == "train":
        train(env_id=args.env_id, total_timesteps=args.timesteps)
    else:
        enjoy(env_id=args.env_id, model_path=args.model_path)

