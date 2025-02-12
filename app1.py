# remaining 
# 1.during no detection of YOLO look for changing of color and crop that part
# 2.during changing of scene in video restart all averaging value

import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

def load_yolo_model(model_path):
    return YOLO(model_path)

def get_video_properties(video_capture):
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    number_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height,number_frames

def resize_frame(frame, target_size=720):
    h, w, _ = frame.shape
    if h < w:
        new_w, new_h = (target_size, int(target_size * (h / w)))
    else:
        new_w, new_h = (int(target_size * (w / h)), target_size)
    return cv2.resize(frame, (new_w, new_h)), new_w, new_h

def get_main_object_box(detections):
    boxes = []
    priorities = []
    for detection in detections:
        for box in detection.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            class_id = int(box.cls[0])
            boxes.append([x1, y1, x2, y2])
            priorities.append(class_id)
    return boxes, priorities

def sort_by_priority(boxes_list, priorities_list):
    for i in range(len(priorities_list)):
        sorted_indices = np.argsort(priorities_list[i])
        priorities_list[i] = np.array(priorities_list[i])[sorted_indices]
        primary_class = priorities_list[i][0]
        boxes_list[i] = np.array(boxes_list[i])[sorted_indices]
        filter_mask = [cls == primary_class for cls in priorities_list[i]]
        boxes_list[i] = boxes_list[i][filter_mask]
    return boxes_list

def get_largest_box(boxes_list):
    return max(boxes_list, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), default=[0, 0, 1, 1])

def apply_rolling_average(values, window=10):
    df = pd.DataFrame({'values': values})
    df['rolling_avg'] = df['values'].rolling(window=window, min_periods=1).mean()
    return df['rolling_avg'].tolist()

def process_video(video_path, model, output_path, stop_threshold=1000):

    cap = cv2.VideoCapture(video_path)
    width, height,max_num_frame = get_video_properties(cap)
    processed_boxes = []
    detection_status = []
    class_priorities = []
    
    frame_count = 0
    while frame_count < stop_threshold:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        resized_frame, new_w, new_h = resize_frame(frame)
        detections = model.track(resized_frame, persist=True)
        boxes, priorities = get_main_object_box(detections)
        
        if not boxes:
            mid_y = new_w // 2
            desired_w = int(9/16 * new_h)
            left_crop, right_crop = mid_y - desired_w // 2, mid_y + desired_w // 2
            processed_boxes.append([[0, left_crop, 0, right_crop]])
            detection_status.append(False)
            class_priorities.append([-1])
        else:
            processed_boxes.append(boxes)
            detection_status.append(True)
            class_priorities.append(priorities)
        
        cv2.imshow("Tracking", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    processed_boxes = sort_by_priority(processed_boxes, class_priorities)
    
    if not detection_status[0]:
        processed_boxes[0] = [[0, 0, new_w - 1, 1]]
    
    for i in range(1, len(processed_boxes)):
        if not detection_status[i]:
            processed_boxes[i] = processed_boxes[i - 1]
    
    largest_boxes = [get_largest_box(box_list) for box_list in processed_boxes]
    
    mid_x_positions = [int((box[0] + box[2]) / 2) for box in largest_boxes]
    smooth_positions = apply_rolling_average(mid_x_positions)
    
    return smooth_positions, new_w, new_h

def save_cropped_video(video_path, smooth_positions, output_path, new_w, new_h, stop_threshold=1000):
    cap = cv2.VideoCapture(video_path)
    desired_w = int(9/16 * new_h)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_video = cv2.VideoWriter(output_path, fourcc, 24, (new_w + desired_w + 10, new_h))
    
    frame_count = 0
    while frame_count < stop_threshold:
        success, frame = cap.read()
        if not success:
            break
        
        resized_frame, new_w, new_h = resize_frame(frame)
        left_crop, right_crop = int(smooth_positions[frame_count] - desired_w // 2), int(smooth_positions[frame_count] + desired_w // 2)
        
        cv2.line(resized_frame, (left_crop, 0), (left_crop, new_h), (0, 0, 255), 2)
        cv2.line(resized_frame, (right_crop, 0), (right_crop, new_h), (0, 0, 255), 2)
        
        cropped_frame = resized_frame[:, left_crop:right_crop]
        output_frame = np.hstack((resized_frame, cropped_frame))
        output_video.write(cv2.resize(output_frame, (new_w + desired_w + 10, new_h)))
        
        cv2.imshow("Cropped", cropped_frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    output_path = f"out_app1.avi"
    video_path = 'test_c/test_c4.mp4'
    
    model = load_yolo_model("yolo11x.pt")
    smooth_positions, new_w, new_h = process_video(video_path, model, output_path)
    save_cropped_video(video_path, smooth_positions, output_path, new_w, new_h)
