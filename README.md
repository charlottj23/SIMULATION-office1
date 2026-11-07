# MuJoCo 기반 강화학습 시뮬레이터 (`CustomRobot-v0`)

커스텀 **MuJoCo MJCF(XML)** 물리 모델, **Gymnasium** 표준 환경 래퍼, **Stable-Baselines3(PPO)** 학습 파이프라인 및 **Hugging Face Hub** 자동 업로드(`huggingface-sb3`)를 포함한 강화학습 시뮬레이터 프로젝트입니다.

---

## 프로젝트 구조

```text
SIMULATION/
├── assets/
│   └── custom_robot.xml        # 커스텀 2자유도(2-DoF) 로봇 팔 MJCF XML 모델
├── envs/
│   ├── __init__.py             # Gymnasium 환경 등록 (CustomRobot-v0)
│   └── custom_mujoco_env.py    # 커스텀 MuJoCo Gymnasium 환경 클래스
├── mujoco_ref/                 # 공식 google-deepmind/mujoco 저장소 (model/ 예제 참조용)
├── train.py                    # PPO 학습, 리플레이 영상(.mp4) 녹화, 3D 인터랙티브 뷰어 실행
├── push_to_hub.py              # 학습된 모델/영상/XML을 Hugging Face Hub에 업로드
└── requirements.txt            # 필수 패키지 목록
```

---

## 실행 방법

### 1. 가상환경 활성화
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. 강화학습(PPO) 실행 및 리플레이 영상 녹화
```powershell
python train.py --mode train --timesteps 50000
```
- 학습된 모델은 `checkpoints/ppo_custom_robot.zip`에 저장됩니다.
- 시뮬레이션 리플레이 영상(`.mp4`)은 `videos/` 폴더에 자동 생성됩니다.

### 3. 3D 인터랙티브 MuJoCo 뷰어로 확인
```powershell
python train.py --mode enjoy --model-path checkpoints/ppo_custom_robot.zip
```

### 4. Hugging Face Hub에 모델 및 리플레이 영상 업로드
```powershell
huggingface-cli login
python push_to_hub.py --repo-id <본인_HF_아이디>/ppo-custom-mujoco-robot
```
- `package_to_hub()`를 통해 모델 가중치, 평가 점수, 리플레이 영상, Model Card(`README.md`), 그리고 커스텀 `assets/custom_robot.xml`이 함께 업로드됩니다.

