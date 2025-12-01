import mujoco 
import numpy as np

if __name__ == "__main__":
    # main()

    # 모델과 데이터 로드
    xml_path = './unitree_go2/go2_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path) # go2.xml 파일 경로
    data = mujoco.MjData(model)

    # 2. (선택 사항) 로봇을 특정 자세로 초기화하고 임의의 속도 부여
    # XML에 정의된 'home' keyframe 자세를 사용합니다.
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key_id != -1:
        data.qpos = model.key_qpos[key_id]

    # Coriolis 효과를 확인하기 위해 0이 아닌 임의의 작은 속도를 설정합니다.
    np.random.seed(0) # 결과를 동일하게 보기 위해 시드 고정
    data.qvel = np.random.randn(model.nv) * 0.1

    # 3. ✨ 가장 중요한 단계: 물리 상태 업데이트 ✨
    # 이 함수가 M, C, g 계산에 필요한 모든 내부 변수를 계산합니다.
    mujoco.mj_forward(model, data)


    # --- 이제 M, C, g를 다시 계산합니다 ---

    # 4. Mass Matrix M 구하기
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    print("Mass Matrix M:\n", M)
    # 이제 M 행렬에 제대로 된 값이 출력될 것입니다.

    # 5. Bias Torque (C*q_dot + g) 구하기
    # 원하는 가속도를 0으로 설정
    data.qacc = np.zeros(model.nv)
    mujoco.mj_inverse(model, data)
    bias_torque = data.qfrc_inverse.copy()
    print("\nBias Torque (C*q_dot + g):\n", bias_torque)

    # 6. Gravity Vector (g)만 구하기
    qvel_backup = data.qvel.copy() # 원래 속도 백업
    data.qvel = np.zeros(model.nv) # 속도를 0으로 만들어 Coriolis 항 제거

    # 속도 값을 바꿨으므로, 다시 mj_forward()를 호출하여 내부 상태를 업데이트해야 합니다.
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data) # qacc는 여전히 0
    gravity_vector = data.qfrc_inverse.copy()

    data.qvel = qvel_backup # 원래 속도로 복원
    print("\nGravity Vector (g):\n", gravity_vector)


    # 7. Coriolis Vector (C*q_dot)만 구하기
    coriolis_vector = bias_torque - gravity_vector
    print("\nCoriolis Vector (C*q_dot):\n", coriolis_vector)
    # data.qvel에 0이 아닌 값을 넣었다면, 이제 이 벡터에도 0이 아닌 값이 나타납니다.






    