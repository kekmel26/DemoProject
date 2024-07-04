import cv2
import argparse

from ultralytics import YOLO
import supervision as sv

from matplotlib.path import Path
import numpy as np

ZONE_POLYGON_LEFT = np.array([
    [0, 0],
    [1280 // 2, 0],
    [1280 // 2, 720],
    [0, 720]
])

ZONE_POLYGON_RIGHT = np.array([
    [720, 0],
    [1280 , 0],
    [1280 , 1280],
    [720, 1280]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def is_point_in_polygon(point, polygon):
    path = Path(polygon)
    return path.contains_point(point)

def main():
    print("Code is started")
    args = parse_arguments() 
    frame_width, frame_height = args.webcam_resolution
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    model = YOLO("yolov8n.pt")
    print("Model is loaded")

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator= sv.LabelAnnotator(text_thickness=2, text_scale=1)
    
    zone_left = sv.PolygonZone(polygon= ZONE_POLYGON_LEFT, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_left_annotator = sv.PolygonZoneAnnotator(zone=zone_left, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)

    zone_right = sv.PolygonZone(polygon= ZONE_POLYGON_RIGHT, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_right_annotator = sv.PolygonZoneAnnotator(zone=zone_right, color=sv.Color.GREEN, thickness=2, text_thickness=4, text_scale=2)

    while True:
        ret, frame = cap.read()

        result = model(frame)[0]
        
        det = sv.Detections.from_ultralytics(result)
        #detections = detections[detections.class_id!=0]
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for xyxy, mask, confidence, class_id, trace_id, data in det
        ]

        frame = bounding_box_annotator.annotate(scene=frame, detections=det)
        frame = label_annotator.annotate(scene=frame, detections=det, labels=labels)
        
        zone_left.trigger(detections=det)
        zone_right.trigger(detections=det)

        frame = zone_left_annotator.annotate(scene=frame)
        frame = zone_right_annotator.annotate(scene=frame)


        # Count objects in each zone
        left_objects = sum(is_point_in_polygon((det.xyxy[0], det.xyxy[1]), ZONE_POLYGON_LEFT) for det in det)
        right_objects = sum(is_point_in_polygon((det.xyxy[0], det.xyxy[1]), ZONE_POLYGON_RIGHT) for det in det)

        # Output the message
        if left_objects > 0:
            print(f"There are {left_objects} objects on your left")
        if right_objects > 0:
            print(f"There are {right_objects} objects on your right")

        cv2.imshow("yolov8", frame)

        if(cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()
