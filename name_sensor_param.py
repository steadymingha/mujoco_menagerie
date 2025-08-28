from mujoco import MjModel, MjData
import mujoco.viewer

xml_name = 'unitree_go2/scene_mjx.xml'
model = MjModel.from_xml_path(xml_name)
data = MjData(model)

sensor_name = "gyro"
sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
dim = model.sensor_dim[sensor_id]
addr = model.sensor_adr[sensor_id]

sensor_values = data.sensordata[addr : addr + dim]
print(f"{sensor_name} =", sensor_values)

sensor_info = []

for i in range(model.nsensor):
    adr = model.name_sensoradr[i]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    start = model.sensor_adr[i]
    dim = model.sensor_dim[i]
    sensor_info.append((name, start, dim))

for _ in range(10):
    mujoco.mj_step(model, data)
    for name, start, dim in sensor_info:
        value = data.sensordata[start:start+dim]
        print(f"{name}: {value}")
    print("----")

print(model.opt.timestep)





# if __name__ == "__main__":
#     main()