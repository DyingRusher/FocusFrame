# remaining
# 1.give 102 points to person and 101 to living things
# 2.if a object stayes fixes decrease there points by 3% per frame
# 3. add 1 point to biggest area object
# 4. add 1 to most center object


import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import math


model = YOLO("yolo11x.pt")


video_path = "test_s3.mp4"
cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))


if frame_height < frame_width:
    new_w, new_h = 720, int(720 * (frame_height / frame_width))
else:
    new_w, new_h = int(720 * (frame_width / frame_height)), 720

desired_w = int(9 / 16 * new_h)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_v = cv2.VideoWriter('test_s3_2.avi', fourcc, 24, (new_w + desired_w + 10, new_h))


all_frames = []
ids_score = {}
id_buffer = []
max_id = 0
stop_threshold = 200
frame_count, detection_count, no_detection_count = 0, 0, 0


while frame_count < stop_threshold:
    print(frame_count)
    frame_count += 1
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    frame_data = {'frame_num': frame_count, 'boxes': [], 'classes': [], 'id': [], 'area': [], 'center': [], 'num': 0}

    if len(results[0].boxes.xyxyn) == 0 and results[0].boxes.id is not None:
        detection_count += 1
        for box, cls, obj_id in zip(results[0].boxes.xyxyn, results[0].boxes.cls, results[0].boxes.id):
            frame_data['boxes'].append(box.tolist())
            frame_data['classes'].append(int(cls.item()))
            frame_data['id'].append(int(obj_id.item()))
            frame_data['area'].append(int((box[2] - box[0]) * (box[3] - box[1]) * frame_height * frame_width))
            frame_data['center'].append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            max_id = max(max_id, int(obj_id.item()))
        frame_data['num'] = len(results[0].boxes.xyxyn)
    else:
        no_detection_count += 1
    
    all_frames.append(frame_data)
    cv2.imshow("YOLO Tracking", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ids_score = {i: 100 for i in range(max_id + 1)}
for frame in all_frames[:4]:
    id_buffer.append(frame['id'])
    for obj_id in frame['id']:
        if obj_id not in ids_score:
            ids_score[obj_id] = 100

mid_y_list = [new_w // 2] * 4
for i in range(4, len(all_frames)):
    num_objects = all_frames[i]['num']
    if num_objects > 0:
        ids_now = all_frames[i]['id']
        dis_list = []

        for obj_index, obj_id in enumerate(ids_now):
            for j in range(1, 5):
                if obj_id in id_buffer[-j]:
                    prev_index = id_buffer[-j].index(obj_id)
                    init_point = all_frames[i - j]['center'][prev_index]
                    final_point = all_frames[i]['center'][obj_index]

                    init_x, init_y = int(init_point[0] * frame_width), int(init_point[1] * frame_height)
                    final_x, final_y = int(final_point[0] * frame_width), int(final_point[1] * frame_height)

                    dis_travel = math.sqrt((init_x - final_x) ** 2 + (init_y - final_y) ** 2) / j
                    dis_list.append(0 if dis_travel < 5 else dis_travel)
                    break
                else:
                    dis_list.append(0)
                    break

        id_buffer.pop(0)
        id_buffer.append(ids_now)
        dis_list = np.array(dis_list)
        sorted_indices = np.argsort(dis_list)[::-1]

        score_increment = np.linspace(3, 0, num_objects).astype(int)
        for rank, obj_index in enumerate(sorted_indices):
            obj_id = ids_now[obj_index]
            ids_score[obj_id] = min(200, ids_score[obj_id] + (ids_score[obj_id] * score_increment[rank]) / 100)

        max_score_index = np.argmax([ids_score[obj_id] for obj_id in ids_now])
        mid_y = int(all_frames[i]['center'][max_score_index][0] * new_w)
        mid_y = max(desired_w // 2 + 1, min(mid_y, new_w - desired_w // 2 - 1))
        mid_y_list.append(mid_y)
    else:
        id_buffer.pop(0)
        id_buffer.append([])
        mid_y_list.append(new_w // 2)

mid_y_df = pd.DataFrame({'values': mid_y_list})
mid_y_df['rolling_avg'] = mid_y_df['values'].rolling(window=10).mean().bfill()
mid_y_list = mid_y_df['rolling_avg'].tolist()

cap2 = cv2.VideoCapture(video_path)
frame_count = 0
while frame_count < stop_threshold:
    success, frame = cap2.read()
    if not success:
        break

    frame = cv2.resize(frame, (new_w, new_h))
    left_crop, right_crop = int(mid_y_list[frame_count] - desired_w // 2), int(mid_y_list[frame_count] + desired_w // 2)
    cropped_frame = frame[:, left_crop:right_crop]
    combined_frame = np.hstack((frame, cropped_frame))

    cv2.line(frame, (left_crop, 0), (left_crop, new_h), (0, 0, 255), 1)
    cv2.line(frame, (right_crop, 0), (right_crop, new_h), (0, 0, 255), 1)
    
    cv2.imshow("Cropped", cropped_frame)
    out_v.write(cv2.resize(combined_frame, (new_w + desired_w + 10, new_h)))
    cv2.imshow("Original", frame)
    cv2.waitKey(1)
    frame_count += 1

cap.release()
cap2.release()
cv2.destroyAllWindows()
