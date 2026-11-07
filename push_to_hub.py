import argparse
from pathlib import Path

import gymnasium as gym
from huggingface_hub import HfApi
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import envs  # Register CustomRobot-v0 and CustomWorld-v0


def push_model_to_hub(
    token: str,
    repo_name: str = "mujoco-office-tidy-robot-ppo",
    model_path: str = "checkpoints/ppo_world_robot.zip",
):
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{repo_name}"

    print(f"[1/4] Hugging Face 계정 확인됨: {username}")
    print(f"[2/4] 저장소 생성/확인 중: https://huggingface.co/{repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # 모델 평가 점수 산출
    env = gym.make("CustomWorld-v0")
    model = PPO.load(model_path, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    env.close()

    # Model Card (README.md) 생성 (허깅페이스 웹페이지에서 바로 replay.mp4가 재생되도록 구성)
    model_card = f"""---
library_name: stable-baselines3
tags:
- mujoco
- reinforcement-learning
- stable-baselines3
- ppo
- robotics
- mobile-manipulator
- physical-ai
model-index:
- name: PPO
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: CustomWorld-v0
      type: CustomWorld-v0
    metrics:
    - type: mean_reward
      value: {mean_reward:.2f} +/- {std_reward:.2f}
      name: mean_reward
---

# 🦾 MuJoCo Office Tidy Mobile Manipulator (`CustomWorld-v0`) - PPO

**MuJoCo 4.0** 기반의 **사무실 정리/수거 모바일 매니퓰레이터 로봇(`office_tidy_mobile_manipulator`)** 강화학습 모델 및 시뮬레이터 저장소입니다.

## 🎥 Simulation Replay
<video controls autoplay loop muted playsinline width="640">
  <source src="https://huggingface.co/{repo_id}/resolve/main/replay.mp4" type="video/mp4">
</video>

## 🤖 Environment & Robot Specifications
- **Simulator**: MuJoCo (`world.xml`)
- **Robot**: Holonomic Mobile Base (2-DoF) + 4-DoF Articulated Robot Arm + Parallel-Jaw Gripper (Total 7 Actuators)
- **Task**: 사무실 바닥에 떨어진 빈 캔(`soda_can`), 플라스틱 블록(`plastic_block`), 종이컵(`coffee_cup`)을 직접 그리퍼로 파지(Pick)하여 분리수거함(`trash_bin`)에 넣는(Place) 작업
- **Algorithm**: Stable-Baselines3 `PPO` (`MlpPolicy`)
- **Evaluation Reward**: `{mean_reward:.2f} +/- {std_reward:.2f}`

## 🚀 Usage

```bash
pip install -r requirements.txt
# 3D 인터랙티브 MuJoCo 뷰어 실행
python run.py
# 추가 강화학습 실행
python train.py --env-id CustomWorld-v0 --mode train --timesteps 100000
```
"""
    card_path = Path("HF_README.md")
    card_path.write_text(model_card, encoding="utf-8")

    print("[3/4] 모델 가중치, 시뮬레이션 영상(replay.mp4), MuJoCo XML 및 코드 업로드 중...")
    uploads = [
        ("HF_README.md", "README.md"),
        (model_path, "ppo_world_robot.zip"),
        ("checkpoints/best_model/best_model.zip", "ppo_custom_robot_2dof_best.zip"),
        ("videos/office_tidy_manipulator.mp4", "replay.mp4"),
        ("videos/ppo_world_robot-replay-step-0-to-step-200.mp4", "videos/ppo_world_robot_rl_replay.mp4"),
        ("videos/bed_spring_100kg_human.mp4", "videos/bed_spring_100kg_human.mp4"),
        ("videos/bed_spring_50kg_human.mp4", "videos/bed_spring_50kg_human.mp4"),
        ("videos/cement_basketball_bounce.mp4", "videos/cement_basketball_bounce.mp4"),
        ("videos/cement_steel_ball_drop.mp4", "videos/cement_steel_ball_drop.mp4"),
        ("world.xml", "world.xml"),
        ("assets/custom_robot.xml", "assets/custom_robot.xml"),
        ("run.py", "run.py"),
        ("train.py", "train.py"),
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
        ("envs/__init__.py", "envs/__init__.py"),
        ("envs/world_env.py", "envs/world_env.py"),
        ("envs/custom_mujoco_env.py", "envs/custom_mujoco_env.py"),
    ]

    for local_file, path_in_repo in uploads:
        p = Path(local_file)
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  ✓ 업로드 완료: {path_in_repo}")

    if card_path.exists():
        card_path.unlink()

    print(f"\n[4/4] 🎉 Hugging Face Hub 업로드 완료!")
    print(f"👉 모델 저장소 주소: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo 강화학습 모델 및 시뮬레이터를 Hugging Face Hub에 업로드")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Write Token")
    parser.add_argument("--repo-name", type=str, default="mujoco-office-tidy-robot-ppo", help="생성할 저장소 이름")
    args = parser.parse_args()

    push_model_to_hub(token=args.token, repo_name=args.repo_name)
