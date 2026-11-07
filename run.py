import math
from pathlib import Path
import time

import mujoco
from mujoco import viewer
import numpy as np


def move_towards(curr: float, target: float, max_delta: float) -> float:
    if abs(target - curr) <= max_delta:
        return target
    return curr + math.copysign(max_delta, target - curr)


def main():
    xml_path = Path(__file__).resolve().parent / "world.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    trash_names = ["soda_can", "plastic_block", "coffee_cup"]
    trash_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in trash_names
    ]
    bin_pos = np.array([0.0, -0.52, 0.28])

    mujoco.mj_forward(model, data)

    print("=======================================================================")
    print(" 🦾 사무실 정리/수거 모바일 매니퓰레이터 로봇 시뮬레이터 (world.xml)")
    print(" - 로봇청소기 주행 베이스 + 4자유도 다관절 로봇 팔 + 평행 그리퍼(집게)")
    print(" - 바닥에 떨어진 빈 캔(빨강), 플라스틱 블록(파랑), 컵(노랑)을 순서대로")
    print("   직접 집어 올려(Pick) 초록색 분리수거함(Trash Bin)에 넣습니다(Place)!")
    print(" - 물체를 더블클릭 후 Ctrl+드래그로 다른 곳에 옮겨도 로봇이 추적합니다.")
    print("=======================================================================")

    # 포즈 정의: (shoulder, elbow, wrist)
    POSE_HOME = (-0.25, 0.45, 0.20)
    POSE_REACH_HIGH = (0.35, 0.35, 0.15)
    POSE_REACH_LOW = (0.78, 0.50, -0.45)  # 바닥 높이(z~0.045m), 수평 전방 약 0.41m
    POSE_CARRY = (-0.15, 0.35, 0.15)
    POSE_DROP = (0.35, 0.45, -0.10)

    GRIP_OPEN = 0.022
    GRIP_CLOSE = -0.028

    with viewer.launch_passive(model, data) as v:
        v.cam.azimuth = 135
        v.cam.elevation = -26
        v.cam.distance = 2.1
        v.cam.lookat[:] = [0.0, -0.05, 0.22]

        current_item_idx = 0
        phase = 0
        phase_timer = 0.0

        # 현재 명령 상태 초기화
        cmd = np.array([0.0, 0.0, 0.0, *POSE_HOME, GRIP_OPEN], dtype=np.float64)

        while v.is_running():
            step_start = time.time()
            dt = model.opt.timestep
            phase_timer += dt

            b_id = trash_body_ids[current_item_idx]
            obj_pos = data.xpos[b_id].copy()

            # 로봇 베이스를 물체(또는 쓰레기통)로부터 거리 R_STOP = 0.415m 떨어진 위치로 이동
            R_STOP = 0.415

            if phase == 0:
                # 1단계: 목표 쓰레기 앞으로 주행 + 그리퍼 열기 + 팔 들어올리기
                vec = obj_pos[:2]
                norm = float(np.linalg.norm(vec))
                yaw = math.atan2(float(obj_pos[1]), float(obj_pos[0]))
                if norm > 0.05:
                    base_target = obj_pos[:2] - (vec / norm) * R_STOP
                else:
                    base_target = np.array([-R_STOP, 0.0])
                    yaw = 0.0

                target_cmd = [base_target[0], base_target[1], yaw, *POSE_REACH_HIGH, GRIP_OPEN]
                if phase_timer > 1.3:
                    phase = 1
                    phase_timer = 0.0

            elif phase == 1:
                # 2단계: 팔을 바닥 쓰레기 높이로 내려 감싸기
                vec = obj_pos[:2] - data.qpos[:2]
                yaw = math.atan2(float(vec[1]), float(vec[0]))
                base_target = obj_pos[:2] - np.array([math.cos(yaw), math.sin(yaw)]) * R_STOP
                target_cmd = [base_target[0], base_target[1], yaw, *POSE_REACH_LOW, GRIP_OPEN]
                if phase_timer > 1.1:
                    phase = 2
                    phase_timer = 0.0

            elif phase == 2:
                # 3단계: 그리퍼(집게) 꽉 닫아 물체 파지(Grasp)
                target_cmd = [cmd[0], cmd[1], cmd[2], *POSE_REACH_LOW, GRIP_CLOSE]
                if phase_timer > 0.8:
                    phase = 3
                    phase_timer = 0.0

            elif phase == 3:
                # 4단계: 물체를 집은 채로 팔 들어올리기 (Lift)
                target_cmd = [cmd[0], cmd[1], cmd[2], *POSE_CARRY, GRIP_CLOSE]
                if phase_timer > 1.0:
                    phase = 4
                    phase_timer = 0.0

            elif phase == 4:
                # 5단계: 초록색 분리수거함(Trash Bin) 앞으로 이동 및 회전
                vec_bin = bin_pos[:2] - data.qpos[:2]
                yaw_bin = math.atan2(float(bin_pos[1]), float(bin_pos[0]))
                base_target = bin_pos[:2] - np.array([math.cos(yaw_bin), math.sin(yaw_bin)]) * R_STOP
                yaw_to_bin = math.atan2(
                    float(bin_pos[1] - base_target[1]),
                    float(bin_pos[0] - base_target[0]),
                )
                target_cmd = [base_target[0], base_target[1], yaw_to_bin, *POSE_DROP, GRIP_CLOSE]
                if phase_timer > 1.5:
                    phase = 5
                    phase_timer = 0.0

            elif phase == 5:
                # 6단계: 분리수거함 위에서 그리퍼 열어 쓰레기 투하 (Release)
                target_cmd = [cmd[0], cmd[1], cmd[2], *POSE_DROP, GRIP_OPEN]
                if phase_timer > 0.8:
                    print(f"✅ [{trash_names[current_item_idx]}] 수거 완료!")
                    current_item_idx += 1
                    if current_item_idx >= len(trash_body_ids):
                        print("🎉 바닥의 모든 쓰레기(3종)를 분리수거함에 정리했습니다! 다시 리셋합니다.")
                        mujoco.mj_resetData(model, data)
                        mujoco.mj_forward(model, data)
                        current_item_idx = 0
                        cmd[:] = [0.0, 0.0, 0.0, *POSE_HOME, GRIP_OPEN]
                    phase = 0
                    phase_timer = 0.0

            # 부드러운 궤적 보간
            rates = [0.8 * dt, 0.8 * dt, 2.2 * dt, 1.8 * dt, 1.8 * dt, 2.2 * dt, 0.12 * dt]
            for i in range(7):
                cmd[i] = move_towards(float(cmd[i]), float(target_cmd[i]), rates[i])

            data.ctrl[:] = cmd
            mujoco.mj_step(model, data)
            v.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
