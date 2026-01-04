from ultralytics import YOLO
import cv2
import os
import math

class ViolationEngine:
    def __init__(self, model_path, conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf
        os.makedirs("saved_violations", exist_ok=True)

        # Memory to avoid duplicate screenshots
        self.processed_bike_centers = []
        self.MIN_DISTANCE = 80  

    # Code for Bike detection
    def detect_bikes(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        bikes = []

        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if results.names[int(cls)] == "P_Bike":
                    bikes.append(tuple(map(int, box)))

        return bikes

    # code to check rider
    def is_rider(self, person_box, bike_box):
        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bike_box

        # this logic is used to check the feet of rider to differentiate from pedestrians
        foot_x = (px1 + px2) // 2
        foot_y = py2

        return bx1 <= foot_x <= bx2 and by1 <= foot_y <= by2

    # code for deduplication(unique bikes/2wheelers)
    def is_new_bike(self, center):
        for cx, cy in self.processed_bike_centers:
            if math.dist((cx, cy), center) < self.MIN_DISTANCE:
                return False
        return True

    # code for 2 wheeler violations
    def detect_violations_on_bike(self, bike_roi, frame_id, bike_id, global_center):
        results = self.model(bike_roi, conf=self.conf, verbose=False)[0]
        names = results.names

        boxes = []
        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                label = names[int(cls)]
                boxes.append((label, tuple(map(int, box))))

        bike_boxes = [b for l, b in boxes if l == "P_Bike"]
        no_helmet_boxes = [b for l, b in boxes if l == "No_Helmet"]
        mobile_boxes = [b for l, b in boxes if l == "Mobile"]

        violations = []

        # helmet rule
        for nh in no_helmet_boxes:
            for bike in bike_boxes:
                if self.is_rider(nh, bike):
                    violations.append("No Helmet")

        # mobile rule 
        for mob in mobile_boxes:
            for bike in bike_boxes:
                if self.is_rider(mob, bike):
                    violations.append("Mobile Usage")

        # Triple riding 
        if len(bike_boxes) >= 3:
            violations.append("Triple Riding")

        annotated = bike_roi.copy()

       
        if violations:
            for label, (x1, y1, x2, y2) in boxes:
                if label in ["No_Helmet", "Mobile"]:
                    color = (0, 0, 255)
                elif label in ["Helmet", "P_Bike"]:
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 0)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        screenshot_path = None
        if violations and self.is_new_bike(global_center):
            self.processed_bike_centers.append(global_center)
            screenshot_path = f"saved_violations/bike_{bike_id}.jpg"
            cv2.imwrite(screenshot_path, annotated)

        return violations, screenshot_path, annotated
