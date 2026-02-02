import mujoco

# model은 이미 로드되어 있다고 가정
# model = mujoco.MjModel.from_xml_path(xml_path)
xml_path = './unitree_go2/go2_mjx_mh.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
print(f"{'Joint Name':<20} | {'DoF Index (Jacobian/qvel)'}")
print("-" * 50)

# 모든 관절(joint)을 순회하며 정보 출력
for i in range(model.njnt):
    # 관절 이름 가져오기
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    
    if name is None:
        name = "No Name (Base/Freejoint)"
    
    dof_adr = model.jnt_dofadr[i]
    print(f"{name:<30} | {dof_adr}")