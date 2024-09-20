#!/usr/bin/env python3

"""Example module for Hailo YOLOv5 person and face detection."""

import argparse
import cv2
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo
import numpy as np
import json
import signal
import sys

def resize_and_pad(frame, target_size=(640, 640), color=(0, 0, 0)):
    if frame.shape[2] == 4:
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

    return padded_frame

#def extract_detections(hailo_output, w, h, class_names, label_offset=1, threshold=0.5):
def extract_detections(hailo_output, original_w, original_h, resized_w, resized_h, class_names, threshold=0.25):
    """Extract detections from the HailoRT-postprocess output, adjusting for label offset."""
    results = []

    # Calculate the scale factor needed to fit the original image into the resized dimensions
    scale = min(resized_w / original_w, resized_h / original_h)
    # Calculate the padding required to center the original image within the resized image
    pad_w = (resized_w - int(original_w * scale)) // 2
    pad_h = (resized_h - int(original_h * scale)) // 2

    for class_id, detections in enumerate(hailo_output):
        adjusted_class_id = class_id + label_offset
        if adjusted_class_id < len(class_names):
            for detection in detections:
                score = detection[4]
                if score >= threshold:
                    y0, x0, y1, x1 = detection[:4]

                    # Scale the coordinates to match the resized image dimensions
                    x0 = int(x0 * resized_w)
                    y0 = int(y0 * resized_h)
                    x1 = int(x1 * resized_w)
                    y1 = int(y1 * resized_h)

                    # Adjust coordinates to remove the padding and scale them back to original dimensions
                    x0 = max(0, int((x0 - pad_w) / scale))
                    y0 = max(0, int((y0 - pad_h) / scale))
                    x1 = min(original_w, int((x1 - pad_w) / scale))
                    y1 = min(original_h, int((y1 - pad_h) / scale))

                    bbox = (x0, y0, x1, y1)
                    #bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                    results.append([class_names[adjusted_class_id], bbox, score])
    return results


def draw_objects(request):
    current_detections = detections
    if current_detections:
        with MappedArray(request, "main") as m:
            for class_name, bbox, score in current_detections:
                x0, y0, x1, y1 = bbox
                label = f"{class_name} %{int(score * 100)}"
                cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0, 0), 2)
                cv2.putText(m.array, label, (x0 + 5, y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0, 0), 2, cv2.LINE_AA)

def signal_handler(sig, frame):
    global video_writer
    print("Exiting...")
    if video_writer is not None:
        video_writer.release()  # Release the video writer before exiting
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description="YOLOv5 Person and Face Detection Example")
    parser.add_argument("-m", "--model", help="Path for the HEF model.",
                        default="/usr/share/hailo-models/yolov5s_personface_h8l.hef")
    parser.add_argument("-c", "--config", help="Path to the JSON configuration file.",
                        default="/usr/share/hailo-models/yolov5_personface.json")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.5,
                        help="Score threshold, must be a float between 0 and 1.")
    parser.add_argument("-o", "--output", default="face_annotated_output.mp4",
                        help="Path to the output video file.")
    args = parser.parse_args()

    # Load configuration file for anchors, labels, etc.
    with open(args.config, 'r', encoding="utf-8") as f:
        config = json.load(f)

    class_names = config["labels"]
    label_offset = config["label_offset"]
    iou_threshold = config["iou_threshold"]
    detection_threshold = config["detection_threshold"]

    # Get the Hailo model, the input size it wants, and the size of our preview stream.
    with Hailo(args.model) as hailo:
        model_h, model_w, _ = hailo.get_input_shape()
        video_w, video_h = 1920, 1080

        # The list of detected objects to draw.
        detections = None

        # Configure and start Picamera2.
        with Picamera2() as picam2:
            main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            lores = {'size': (1280, 720), 'format': 'RGB888'}
            controls = {'FrameRate': 30}
            config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
            picam2.configure(config)
            # Initialize the VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
            video_writer = cv2.VideoWriter(args.output, fourcc, 12.0, (video_w, video_h))
            picam2.start_preview(Preview.QTGL, x=0, y=0, width=800, height=400)
            picam2.start()
            picam2.pre_callback = draw_objects

            # Process each low resolution camera frame.
            while True:
                frame = picam2.capture_array('lores')
                resized_frame = resize_and_pad(frame, target_size=(640, 640))
                results = hailo.run(resized_frame)

                # Extract detections from the inference results, using the label offset
                detections = extract_detections(results[0], video_w, video_h, 640, 640, class_names, args.score_thresh)
#                detections = extract_detections(results[0], video_w, video_h, class_names, label_offset, detection_threshold)
                frame_rgb = picam2.capture_array('main')
                frame_rgb = frame_rgb[:, :, [0, 1, 2]] 

                video_writer.write(frame_rgb) 
