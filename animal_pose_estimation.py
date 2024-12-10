
import cv2, os
import pandas as pd
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ann_meta_data = pd.read_csv("keypoint_definitions.csv")
COLORS = ann_meta_data["Hex colour"].values.tolist()
COLORS_RGB_MAP = []
for color in COLORS:
    R, G, B = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
    COLORS_RGB_MAP.append({color: (R,G,B)})

# joint = { "LF_foot" : 0, "LF_calf :1, "LF_thight" : 2, "LR_foot": 3, "LR_calf": 4, # no use
joint= {"RF_foot" : 6, "RF_calf" : 7, "RF_thigh" : 8,
        "RR_foot" : 9, "RR_calf" : 10, "RR_thigh" : 11}
# "tail" : 12, "head" : 15, "nose" : 16, "mouse" : 17 # no use

def angle(v1, v2):
    v1Xv2 = np.cross(v1, v2)
    v1Xv2_norm = np.sqrt(np.sum(v1Xv2**2))
    angle = np.arctan2(v1Xv2_norm, v1@v2)

    if v1Xv2 < 0: angle *= -1

    return angle

result = [] # Front right thigh, Front right calf, Rear right thigh, Rear right calf

def cal_angle_from_vector(kpts):

    # extract x,y position from kpts estimated
    right_front_calf_joint  = kpts[kpts[:, 2] == joint["RF_calf"]][0][:2]
    right_front_thigh_joint = kpts[kpts[:, 2] == joint["RF_thigh"]][0][:2]
    right_front_foot        = kpts[kpts[:, 2] == joint["RF_foot"]][0][:2]

    right_rear_calf_joint   = kpts[kpts[:, 2] == joint["RR_calf"]][0][:2]
    right_rear_thigh_joint  = kpts[kpts[:, 2] == joint["RR_thigh"]][0][:2]
    right_rear_foot         = kpts[kpts[:, 2] == joint["RR_foot"]][0][:2]

    # image coordinate to normal cartesian
    conversion_vec = np.array([1, -1])
    right_front_calf_joint *= conversion_vec
    right_front_thigh_joint *= conversion_vec
    right_front_foot *= conversion_vec

    right_rear_calf_joint *= conversion_vec
    right_rear_thigh_joint *= conversion_vec
    right_rear_foot *= conversion_vec

    # cal vector
    front_thigh = right_front_calf_joint - right_front_thigh_joint
    front_calf =  right_front_foot       - right_front_calf_joint
    rear_thigh =  right_rear_calf_joint  - right_rear_thigh_joint
    rear_calf =   right_rear_foot        - right_rear_calf_joint

    # Angle (rad)
    ref_axis = np.array((0, -1))

    right_front_thigh_angle = angle(ref_axis, front_thigh)
    right_front_calf_angle = angle(ref_axis, front_calf)
    right_rear_thigh_angle = angle(ref_axis, rear_thigh)
    right_rear_calf_angle = angle(ref_axis, rear_calf)

    # right_front_thigh_angle = angle(front_thigh, ref_axis)
    # right_front_calf_angle = angle(front_calf, ref_axis)
    # right_rear_thigh_angle = angle(rear_thigh, ref_axis)
    # right_rear_calf_angle = angle(rear_calf, ref_axis)

    result.append([right_front_thigh_angle, right_front_calf_angle, right_rear_thigh_angle, right_rear_calf_angle])

     # result = []  # Front right thigh, Front right calf, Rear right thigh, Rear right calf
    return result

def draw_landmarks(image, landmarks):
    radius = 5
    if (image.shape[1] > 1000):
        radius = 8

    # 선을 그릴 연결 포인트 정의
    connections = [
        (0, 1), (1, 2), (3, 4), (4, 5),
        (6, 7), (7, 8), (9, 10), (10, 11)
    ]

    # 연결된 선 그리기
    for connection in connections:
        point_start, point_end = connection

        # 해당 point 번호가 landmarks에 존재하는지 확인
        start_points = [kpt for kpt in landmarks if kpt[-1] == point_start]
        end_points = [kpt for kpt in landmarks if kpt[-1] == point_end]

        if start_points and end_points:
            # 시작점과 끝점 좌표 가져오기
            start_x, start_y = start_points[0][:2].astype("int").tolist()
            end_x, end_y = end_points[0][:2].astype("int").tolist()

            # 선 그리기 (흰색)
            cv2.line(image,
                     (start_x, start_y),
                     (end_x, end_y),
                     color=(255, 255, 255),
                     thickness=2,
                     lineType=cv2.LINE_AA)

    for idx, kpt_data in enumerate(landmarks):
        loc_x, loc_y = kpt_data[:2].astype("int").tolist()
        color_id = list(COLORS_RGB_MAP[int(kpt_data[-1])].values())[0]

        cv2.circle(image,
                   (loc_x, loc_y),
                   radius,
                   color=color_id[::-1],
                   thickness=-1,
                   lineType=cv2.LINE_AA)

        cv2.putText(
            image,
            str(int(kpt_data[-1])),  # color_id를 텍스트로 변환
            (loc_x - 20, loc_y - radius - 10),  # 텍스트 위치
            cv2.FONT_HERSHEY_SIMPLEX,  # 폰트
            0.6,  # 글자 크기
            (255, 255, 255),  # 글자 색상 (흰색)
            1,  # 글자 두께
            lineType=cv2.LINE_AA  # 선의 부드러움
        )



    return image

def draw_boxes(image, detections, class_name="dog", score=None, color=(0, 255, 0)):
    font_size = 0.25 + 0.07 * min(image.shape[:2]) / 100
    font_size = max(font_size, 0.5)
    font_size = min(font_size, 0.8)
    text_offset = 3

    thickness = 2
    if (image.shape[1] > 1000):
        thickness = 10

    xmin, ymin, xmax, ymax = detections[:4].astype("int").tolist()
    conf = round(float(detections[-1]), 2)
    cv2.rectangle(image,
                  (xmin, ymin),
                  (xmax, ymax),
                  color=(0, 255, 0),
                  thickness=thickness,
                  lineType=cv2.LINE_AA)

    display_text = f"{class_name}"
    if score is not None:
        display_text += f": {score:.2f}"

    (text_width, text_height), _ = cv2.getTextSize(display_text,
                                                   cv2.FONT_HERSHEY_SIMPLEX,
                                                   font_size, 2)

    cv2.rectangle(image,
                  (xmin, ymin),
                  (xmin + text_width + text_offset, ymin - text_height - int(15 * font_size)),
                  color=color, thickness=-1)

    image = cv2.putText(
        image,
        display_text,
        (xmin + text_offset, ymin - int(10 * font_size)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 0, 0),
        2, lineType=cv2.LINE_AA,
    )

    return image

def prepare_predictions(
    image_path,
    model,
    BOX_IOU_THRESH=0.55,
    BOX_CONF_THRESH=0.30,
    KPT_CONF_THRESH=0.68):

    image = cv2.imread(image_path).copy()

    results = model.predict(image_path, conf=BOX_CONF_THRESH, iou=BOX_IOU_THRESH)[0].cpu()

    if not len(results.boxes.xyxy):
        return image

    pred_boxes = results.boxes.xyxy.numpy()
    pred_box_conf = results.boxes.conf.numpy()
    pred_kpts_xy = results.keypoints.xy.numpy()
    pred_kpts_conf = results.keypoints.conf.numpy()

    for boxes, score, kpts, confs in zip(pred_boxes, pred_box_conf, pred_kpts_xy, pred_kpts_conf):
        kpts_ids = np.where(confs > KPT_CONF_THRESH)[0]
        filter_kpts = kpts[kpts_ids]
        filter_kpts = np.concatenate([filter_kpts, np.expand_dims(kpts_ids, axis=-1)], axis=-1)
        image = draw_boxes(image, boxes, score=score)
        image = draw_landmarks(image, filter_kpts)

    cv2.imshow('Window Name', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


def predictions(
    image_fame,
    model,
    BOX_IOU_THRESH=0.55,
    BOX_CONF_THRESH=0.30,
    KPT_CONF_THRESH=0.68):

    image = image_fame.copy()

    results = model.predict(image_fame, conf=BOX_CONF_THRESH, iou=BOX_IOU_THRESH, verbose=False)[0].cpu()

    if not len(results.boxes.xyxy):
        return image

    pred_boxes = results.boxes.xyxy.numpy()
    pred_box_conf = results.boxes.conf.numpy()
    pred_kpts_xy = results.keypoints.xy.numpy()
    pred_kpts_conf = results.keypoints.conf.numpy()

    for boxes, score, kpts, confs in zip(pred_boxes, pred_box_conf, pred_kpts_xy, pred_kpts_conf):
        kpts_ids = np.where(confs > KPT_CONF_THRESH)[0]
        filter_kpts = kpts[kpts_ids]
        filter_kpts = np.concatenate([filter_kpts, np.expand_dims(kpts_ids, axis=-1)], axis=-1)

        image = draw_landmarks(image, filter_kpts)
        angles_inframe = cal_angle_from_vector(filter_kpts)

    # np.save('pace_angles.npy', np.array(angles_inframe))

    # cv2.imshow('dog gait', image)
    # cv2.waitKey(0)
    return image, angles_inframe # image

def video_save(cap, video_path):
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 카메라에 따라 값이 정상적, 비정상적

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    return out



def main():
    # Paths
    model_path = "best.pt"  # Replace with actual model path
    movie_path = "dog_pace.mp4"
    video_path = "{}_pace.mp4".format(movie_path)
    save_flag = True
    frame_max = 50
    label_name = {0: "Front right thigh", 1: "Front right calf", 2: "Rear right thigh", 3: "Rear right calf"}

#### real-time plot ######
    # 플롯 설정
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1행 2열 레이아웃
    plt.subplots_adjust( left=0.05, right=0.95)


    image_ax, graph_ax = ax
    frame_image = None
    angles_plot, = graph_ax.plot([], [], 'bo-', label='Joint Angles')
    plt.ion()  # 인터랙티브 모드 활성화

# ######## real-time plot ######

    model_pose = YOLO(model_path)

    cap = cv2.VideoCapture(movie_path)
    # out = video_save(cap, video_path)

    frame_num = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        if not ret or frame_num == frame_max:
            break

        img, angles = predictions(frame, model_pose)
        # out.write(img)

        # plt.plot(angles[-1], '.-', label=["right_front_thigh_angle", "right_front_calf_angle", "right_rear_thigh_angle", "right_rear_calf_angle"])
        # cv2.imshow('dog gait', img)
        # plt.show()
        # cv2.waitKey(0)

# ### real-time plot, image with graph
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_ax.clear()
        image_ax.imshow(img_rgb)
        image_ax.set_title("Video Frame")


        # Update angle graph
        graph_ax.clear()
        angles_np = np.rad2deg(np.array(angles))
        # for i in range(0, 4):
        #     graph_ax.plot(angles_np[:, i], '.-', label=label_name[i])

        graph_ax.plot(range(len(angles)), angles, '.-', label=["right_front_thigh_angle", "right_front_calf_angle", "right_rear_thigh_angle", "right_rear_calf_angle"])
        # graph_ax.plot(range(angles_np[:,2].shape[0]), angles_np[:,2], '.-', label="right_rear_thigh_angle")
        # graph_ax.set_title("Joint Angles")
        # graph_ax.set_xlabel("Joint Index")
        # graph_ax.set_ylabel("Angle (degrees)")
        graph_ax.set_xlim(0, frame_max)
        # graph_ax.set_ylim(0, 45)  # Set y-axis limits for angles
        graph_ax.grid()

        graph_ax.legend(loc='upper right')

        plt.pause(0.1)  # Pause to update plots
#### END #########################
        Test = 0

        frame_num += 1


    cap.release()
    cv2.destroyAllWindows()

    if save_flag:
        np.save('raw_data_100.npy', np.array(angles))


    plt.ioff()
    plt.show()

    # plt.plot(angles)
    # plt.grid()
    # plt.show()

print("completed!")
if __name__ == "__main__":
    main()
