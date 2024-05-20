import sys
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image
import streamlit as st

ZONE_POLYGON = np.array([[0, 0], [1280 // 2, 0], [1280 // 2, 720], [0, 720]])

class_labels = {
    0: "Apple",
    1: "Banana",
    2: "Grape",
    3: "Orange",
    4: "Pineapple",
    5: "Watermelon",
    # Add more class IDs and labels as needed
}

parser = argparse.ArgumentParser(description="YoloV8 Object Detection")
parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
args = parser.parse_args()

model = YOLO('best_200epochs.pt')

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1,
)

zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=tuple(args.webcam_resolution))
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red(), text_scale=2, text_thickness=4, thickness=2)

def display_artwork(image):
    with st.expander("Picture", expanded=True):
        st.image(image, use_column_width=True)

def detect_fruit():
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        ret, frame = cap.read()

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    pass


def detect_fruit_in_image(image):
    # Make predictions on the image
    result = model(image)[0]
    result_arr = result.orig_img

    # Convert YOLO detections to custom detections
    detections = sv.Detections.from_yolov8(result)

    # Get labels from detections
    labels = [
        f"{model.model.names[class_id]}: {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]

    # Annotate the image with bounding boxes and labels
    detected_img = Image.fromarray(result.plot()[:, :, ::-1])
    detected_img_arr = np.array(detected_img)
    detected_img_rgb = cv2.cvtColor(detected_img_arr, cv2.COLOR_BGR2RGB)
    display_artwork(detected_img)
    #cv2.imshow("Detected Image", detected_img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return labels


def detect_fruit_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        results = model(frame)[0]

        detections = sv.Detections.from_yolov8(results)

        labels = [
            f"{model.model.names[class_id]}: {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

        cap.release()
        cv2.destroyAllWindows()
        pass


def count_fruits_on_camera():
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)


    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        ret, frame = cap.read()

        results = model(frame, agnostic_nms=True)[0]

        detections = sv.Detections.from_yolov8(results)

        labels = [
            f"{model.model.names[class_id]}: {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    pass


def detect_only(class_id):
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        ret, frame = cap.read()

        results = model(frame, agnostic_nms=True)[0]

        detections = sv.Detections.from_yolov8(results)
        detections = detections[detections.class_id == class_id]  # Change to another class_id here

        labels = [
            f"{model.model.names[class_id]}: {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    pass


def format_result(result):
    fruit_count = {}
    for item in result:
        fruit, _ = item.split(':')
        fruit_count[fruit] = fruit_count.get(fruit, 0) + 1

    output = []
    for fruit, count in fruit_count.items():
        output.append(f"{count} {fruit.capitalize() if count == 1 else fruit.capitalize() + 's'}")

    return '\n'.join(output) + "\nin this image"


#if __name__ == "__main__":
    # Hàm 1: detect_fruit_in_image
    #result = detect_fruit_in_image("images/group_apple_2.jpg")

    # print(fruit_detector.format_result(result))

    # Hàm 2: detect_fruit_in_video
    # detect_fruit_in_video("videos/video_2.mp4")

    # Hàm 3: count fruit on camera
    # count_fruits_on_camera()

    # Hàm 4: detect only Apple (0) (you can change to another class_id (1, 2, 3, 4, 5))
    # detect_only(0)