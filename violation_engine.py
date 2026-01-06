from ultralytics import YOLO
import cv2
import os
import math
import easyocr

class ViolationEngine:
    def __init__(self, model_path, conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf 
        
        self.helmet_conf = 0.80
        self.mobile_conf = 0.50
        self.rider_conf = 0.60  
        
        os.makedirs("saved_violations", exist_ok=True)
        self.processed_bike_centers = []
        self.MIN_DISTANCE = 80 
        
        print("Initializing EasyOCR...") 
        self.reader = easyocr.Reader(['en'], gpu=True) 

    def is_rider(self, person_box, bike_box):
        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bike_box
        foot_x = (px1 + px2) // 2
        return bx1 <= foot_x <= bx2

    def is_new_bike(self, center):
        for cx, cy in self.processed_bike_centers:
            if math.dist((cx, cy), center) < self.MIN_DISTANCE:
                return False
        return True
    
    def reset(self):
        self.processed_bike_centers.clear()

    def detect_violations_on_bike(self, bike_roi, frame_id, bike_id, global_center):
        if bike_roi.size == 0: return [], None, bike_roi, None

        # Run model
        results = self.model(bike_roi, conf=self.conf, verbose=False)[0]
        names = results.names
        annotated = bike_roi.copy()

        boxes = []
        if results.boxes is not None:
            xyxys = results.boxes.xyxy
            clss = results.boxes.cls
            confs = results.boxes.conf.cpu().tolist()

            for box, cls, conf in zip(xyxys, clss, confs):
                label = names[int(cls)]
                
                if label == "No_Helmet" and conf < self.helmet_conf: continue
                if label == "Mobile" and conf < self.mobile_conf: continue
                if label == "P_Bike" and conf < self.rider_conf: continue

                coords = tuple(map(int, box))
                boxes.append((label, coords))
                
                # Draw boxes
                x1, y1, x2, y2 = coords
                color = (200, 200, 200) 
                if label == "No_Helmet": color = (0, 0, 255)
                elif label == "LP": color = (255, 165, 0)
                elif label == "P_Bike": color = (255, 255, 0)
                elif label == "Mobile": color = (255, 0, 255)
                
                label_text = f"{label} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        bike_boxes = [b for l, b in boxes if l == "P_Bike"]
        no_helmet_boxes = [b for l, b in boxes if l == "No_Helmet"]
        mobile_boxes = [b for l, b in boxes if l == "Mobile"]
        plate_boxes = [b for l, b in boxes if l == "LP"] 

        violations = []

        # No Helmet
        for nh in no_helmet_boxes:
            if bike_boxes:
                for bike in bike_boxes:
                    if self.is_rider(nh, bike): violations.append("No Helmet")
            else:
                violations.append("No Helmet")

        # Mobile
        for mob in mobile_boxes:
            if bike_boxes:
                for bike in bike_boxes:
                    if self.is_rider(mob, bike): violations.append("Mobile Usage")
            else:
                violations.append("Mobile Usage")

        # Triple Riding (Dependent on P_Bike detection count)
        if len(bike_boxes) >= 3:
            violations.append("Triple Riding")

        violations = list(set(violations))
        
        # OCR LOGIC
        plate_text = None
        screenshot_path = None
        
        if violations and self.is_new_bike(global_center):
            self.processed_bike_centers.append(global_center)
            screenshot_path = f"saved_violations/bike_{bike_id}.jpg"
            
            if plate_boxes:
                px1, py1, px2, py2 = plate_boxes[0]
                h, w = bike_roi.shape[:2]
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(w, px2), min(h, py2)
                
                plate_crop = bike_roi[py1:py2, px1:px2]
                
                if plate_crop.size > 0:
                    try:
                        ocr_result = self.reader.readtext(plate_crop)
                        extracted_text = " ".join([res[1] for res in ocr_result])
                        clean_text = ''.join(e for e in extracted_text if e.isalnum())
                        
                        if clean_text:
                            plate_text = clean_text.upper()
                            cv2.putText(annotated, f"Plate: {plate_text}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    except Exception as e:
                        print(f"OCR Error: {e}")

            cv2.imwrite(screenshot_path, annotated)
            return violations, screenshot_path, annotated, plate_text

        return violations, None, annotated, None