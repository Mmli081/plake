from ultralytics import YOLO
import cv2
from sort.sort import *
from utils import get_car, write_csv, read_license_plate


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("./best.pt")

cap = cv2.VideoCapture("CmKdPEUOMf2.mp4")

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        if detections_:
            # track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            license_plates = license_plate_detector(frame)[0]
            if license_plates.obb is not None:
                for license_plate in license_plates.obb.xyxy.tolist():
                    x1, y1, x2, y2 = license_plate[
                        :4
                    ]  # Extract x1, y1, x2, y2 from the xyxy tensor
                    score = float(license_plates.obb.conf[0])  # Confidence score
                    class_id = int(license_plates.obb.cls[0])

                    # assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                        license_plate, track_ids
                    )

                    if car_id != -1:

                        # crop license plate
                        license_plate_crop = frame[
                            int(y1) : int(y2), int(x1) : int(x2), :
                        ]

                        # process license plate
                        license_plate_crop_gray = cv2.cvtColor(
                            license_plate_crop, cv2.COLOR_BGR2GRAY
                        )
                        _, license_plate_crop_thresh = cv2.threshold(
                            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                        )

                        # read license plate number
                        license_plate_text, license_plate_text_score = (
                            read_license_plate(license_plate_crop_thresh)
                        )

                        if license_plate_text is not None:
                            results[frame_nmr][car_id] = {
                                "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                                "license_plate": {
                                    "bbox": [x1, y1, x2, y2],
                                    "text": license_plate_text,
                                    "bbox_score": score,
                                    "text_score": license_plate_text_score,
                                },
                            }

write_csv(results, "./test.csv")
