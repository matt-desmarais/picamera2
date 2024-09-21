#!/usr/bin/env python3

import argparse
import cv2
from pose_utils import postproc_yolov8_pose
import numpy as np
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo
from picamera2.encoders import H264Encoder
import signal
import sys

# Joint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, \
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

# Joint pairs for drawing lines between keypoints
JOINT_PAIRS = [[NOSE, L_EYE], [L_EYE, L_EAR], [NOSE, R_EYE], [R_EYE, R_EAR],
               [L_SHOULDER, R_SHOULDER],
               [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST], [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST],
               [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP], [L_HIP, R_HIP],
               [L_HIP, L_KNEE], [R_HIP, R_KNEE], [L_KNEE, L_ANKLE], [R_KNEE, R_ANKLE]]

def resize_and_pad(frame, target_size=(640, 640), color=(0, 0, 0)):
    """Resize and pad the image to maintain aspect ratio."""
    if frame.shape[2] == 4:  # Convert RGBA to RGB if necessary
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    h, w, _ = frame.shape
    target_w, target_h = target_size

    # Calculate aspect ratio and scale the frame
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a new image with the target size and padding
    padded_frame = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

    return padded_frame, x_offset, y_offset, new_w, new_h


def scale_coord(coord, x_offset, y_offset, resized_w, resized_h, original_w, original_h):
    """
    Scale coordinates considering the padding offsets and resized frame.
    Adjust the coordinates back to the original image size (before padding).
    """
    # Calculate the scale factor used for resizing
    scale = min(resized_w / original_w, resized_h / original_h)

    # Calculate padding added to the image
    pad_w = (resized_w - int(original_w * scale)) // 2
    pad_h = (resized_h - int(original_h * scale)) // 2

    # Adjust the coordinates based on padding and scaling
    adj_x = (coord[0] - pad_w - x_offset) / scale
    adj_y = (coord[1] - pad_h - y_offset) / scale

    # Ensure the coordinates are within bounds of the original image
    adj_x = max(0, min(original_w, int(adj_x)))
    adj_y = max(0, min(original_h, int(adj_y)))

    return int(adj_x), int(adj_y)


def visualize_pose_estimation_result(results, image, model_size, detection_threshold=0.5, joint_threshold=0.5, x_offset=0, y_offset=0, new_w=640, new_h=640):
    """Visualize pose estimation results on the image."""
    original_w, original_h = image.shape[1], image.shape[0]

    bboxes, scores, keypoints, joint_scores = (
        results['bboxes'], results['scores'], results['keypoints'], results['joint_scores'])
    box, score, keypoint, keypoint_score = bboxes[0], scores[0], keypoints[0], joint_scores[0]

    for detection_box, detection_score, detection_keypoints, detection_keypoints_score in (
            zip(box, score, keypoint, keypoint_score)):
        if detection_score < detection_threshold:
            continue

        # Draw bounding box
        coord_min = scale_coord(detection_box[:2], x_offset, y_offset, new_w, new_h, original_w, original_h)
        coord_max = scale_coord(detection_box[2:], x_offset, y_offset, new_w, new_h, original_w, original_h)
        cv2.rectangle(image, coord_min, coord_max, (255, 0, 0), 1)
        cv2.putText(image, str(detection_score), coord_min, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        # Draw keypoints
        joint_visible = detection_keypoints_score > joint_threshold
        detection_keypoints = detection_keypoints.reshape(17, 2)
        for joint, joint_score in zip(detection_keypoints, detection_keypoints_score):
            if joint_score > joint_threshold:
                cv2.circle(image, scale_coord(joint, x_offset, y_offset, new_w, new_h, original_w, original_h), 4, (255, 0, 255), -1)

        # Draw lines connecting keypoints
        for joint0, joint1 in JOINT_PAIRS:
            if joint_visible[joint0] and joint_visible[joint1]:
                cv2.line(image, scale_coord(detection_keypoints[joint0], x_offset, y_offset, new_w, new_h, original_w, original_h),
                         scale_coord(detection_keypoints[joint1], x_offset, y_offset, new_w, new_h, original_w, original_h), (255, 0, 255), 3)


def draw_predictions(request):
    """Draw the pose estimation results on the frame."""
    with MappedArray(request, 'main') as m:
        predictions = last_predictions
        if predictions:
            visualize_pose_estimation_result(predictions, m.array, model_size, x_offset=x_offset, y_offset=y_offset, new_w=new_w, new_h=new_h)

def signal_handler(sig, frame):
    global video_writer
    print("Exiting...")
    if video_writer is not None:
        video_writer.release()  # Release the video writer before exiting
    picam2.stop_recording()
    picam2.stop()
    sys.exit(0)


# ---------------- Start of the example --------------------- #

last_predictions = None
video_writer = None  # Global for video writer

if __name__ == "__main__":
    # Define argument parser for model file
    parser = argparse.ArgumentParser(description='Pose estimation using Hailo')
    parser.add_argument('-m', '--model', help="HEF file path", default="/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef")
    parser.add_argument("-r", "--record", default="No", help="Hq or Lq")
    parser.add_argument("-o", "--output", default="pose_annotated_output.mp4", help="Path to the output video file.")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal_handler)


    with Hailo(args.model) as hailo:
        video_w, video_h = (1920, 1080)
        model_h, model_w, _ = hailo.get_input_shape()
        model_size = (model_w, model_h)
        lores_size = (1280, 720)

        with Picamera2() as picam2:
            main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            lores = {'size': lores_size, 'format': 'RGB888'}
            controls = {'FrameRate': 30}
            config = picam2.create_video_configuration(main, lores=lores, controls=controls)
            picam2.configure(config)
            picam2.start_preview(Preview.QTGL, x=0, y=0, width=800, height=400)
            # Initialize the VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
            video_writer = cv2.VideoWriter(args.output, fourcc, 8.5, (video_w, video_h))
            if(args.record == "Lq"):
                # Initialize the VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
                video_writer = cv2.VideoWriter(args.output, fourcc, 12.0, (video_w, video_h))
            elif(args.record == "Hq"):
                encoder = H264Encoder(bitrate=10000000)
                output = args.output
                picam2.start_recording(encoder, output)
            picam2.start()
            picam2.pre_callback = draw_predictions

            while True:
                frame = picam2.capture_array('lores')

                # Resize and pad the frame to 640x640 while keeping the original aspect ratio
                resized_frame, x_offset, y_offset, new_w, new_h = resize_and_pad(frame, target_size=(640, 640))

                # Run inference on the resized and padded frame
                raw_detections = hailo.run(resized_frame)

                # Process the predictions
                last_predictions = postproc_yolov8_pose(1, raw_detections, model_size)
                if(args.record == "Lq"):
                    frame_rgb = picam2.capture_array('main')
                    frame_rgb = frame_rgb[:, :, [0, 1, 2]] 
                    video_writer.write(frame_rgb) 
