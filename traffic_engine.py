import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from violation_engine import ViolationEngine

class TrafficLightManager:
    def __init__(self, traffic_model_path, violation_model_path):
        self.model = YOLO(traffic_model_path)
        self.violation_engine = ViolationEngine(violation_model_path, conf=0.25)
        
        self.track_history = defaultdict(lambda: deque(maxlen=40))
        self.vehicle_directions = {} 
        self.verified_moving = set()
        self.unique_vehicle_classes = {}
        
        # --- FIX 1: Add a set to remember IDs of violators ---
        self.violated_track_ids = set()
        # -----------------------------------------------------

        self.violation_stats = {
            "No Helmet": 0, "Mobile Usage": 0, "Triple Riding": 0, "Total Violations": 0
        }
        
        # Stores dicts: {'image', 'plate', 'violations'}
        self.recent_violations = deque(maxlen=8) 
        self.debug_crops = deque(maxlen=8) 
        
        self.main_lanes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.lane_wait_timers = {d: 0.0 for d in self.main_lanes}
        
        self.weights = {
            'Bus': 2.5, 'Truck': 2.5, 'Mini-bus': 2.0, 'Car': 1.0, 'SUV': 1.0, 
            'Sedan': 1.0, 'Hatchback': 1.0, 'Van': 1.0, 'MUV': 1.0, 
            'Tempo-traveller': 1.5, 'Three-wheeler': 0.8, 'Two-wheeler': 0.5, 
            'Bicycle': 0.5, 'Motorcycle': 0.5, 'Scooter': 0.5
        }
        
        self.movement_threshold = 5 
        self.conf_threshold = 0.1 

    def reset(self):
        self.track_history.clear()
        self.vehicle_directions.clear()
        self.verified_moving.clear()
        self.unique_vehicle_classes.clear()
        
        # --- FIX 2: Clear the memory on reset ---
        self.violated_track_ids.clear()
        # ----------------------------------------
        
        self.lane_wait_timers = {d: 0.0 for d in self.main_lanes}
        self.violation_stats = {"No Helmet": 0, "Mobile Usage": 0, "Triple Riding": 0, "Total Violations": 0}
        self.recent_violations.clear()
        self.debug_crops.clear()
        self.violation_engine.reset()

    def get_detailed_direction(self, track_id, current_x, current_y):
        if track_id not in self.track_history or len(self.track_history[track_id]) < 10: return None
        prev_x, prev_y = self.track_history[track_id][0]
        dx, dy = current_x - prev_x, current_y - prev_y
        if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold: return None 
        if abs(dx) > self.movement_threshold and abs(dy) > self.movement_threshold:
            return f"{'DOWN' if dy > 0 else 'UP'}-{'RIGHT' if dx > 0 else 'LEFT'}"
        return ('DOWN' if dy > 0 else 'UP') if abs(dy) > abs(dx) else ('RIGHT' if dx > 0 else 'LEFT')

    def is_vehicle_stopped(self, track_id):
        history = self.track_history[track_id]
        if len(history) < 20: return False
        start_x, start_y = history[0]
        end_x, end_y = history[-1]
        return np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) < 10 

    def process_frame(self, frame, frame_count):
        results = self.model.track(frame, persist=True, verbose=False, conf=0.1, tracker="botsort.yaml")
        
        raw_moving_density = defaultdict(float)
        raw_stopped_density = defaultdict(float)
        vehicle_counts = defaultdict(int)
        
        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2]
        
        BIKE_CLASSES = ['Two-wheeler', 'Motorcycle', 'Scooter', 'Bike', 'motorcycle', 'scooter', 'motorbike'] 

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            names = results[0].names
            confs = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                x, y, w, h = box
                class_name = names[cls]
                weight = self.weights.get(class_name, 1.0)
                
                # Debug White Box
                rx1, ry1 = int(x - w/2), int(y - h/2)
                rx2, ry2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"{class_name}:{conf:.2f}", (rx1, ry1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                self.unique_vehicle_classes[track_id] = class_name
                vehicle_counts[class_name] += 1
                self.track_history[track_id].append((float(x), float(y)))
                
                start_x, start_y = self.track_history[track_id][0]
                if np.sqrt((x - start_x)**2 + (y - start_y)**2) > 30:
                    self.verified_moving.add(track_id)
                
                raw_direction = self.get_detailed_direction(track_id, x, y)
                stopped = self.is_vehicle_stopped(track_id)
                
                if raw_direction: self.vehicle_directions[track_id] = raw_direction
                final_direction = raw_direction if raw_direction else (self.vehicle_directions.get(track_id) if track_id in self.verified_moving else None)
                if final_direction is None and len(self.track_history[track_id]) > 5: final_direction = "STATIONARY"

                if final_direction:
                    label = final_direction
                    color = (0, 0, 255) if (stopped or final_direction == "STATIONARY") else (0, 255, 0)
                    if stopped and final_direction != "STATIONARY": raw_stopped_density[final_direction] += weight
                    elif final_direction != "STATIONARY": raw_moving_density[final_direction] += weight
                    cv2.putText(annotated_frame, label, (int(x), int(y)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- VIOLATION LOGIC ---
                if class_name in BIKE_CLASSES and frame_count % 5 == 0:
                    pad_w, pad_h = int(w * 0.15), int(h * 0.25)
                    x1, y1 = max(0, int(x-w/2)-pad_w), max(0, int(y-h/2)-pad_h)
                    x2, y2 = min(w_img, int(x+w/2)+pad_w), min(h_img, int(y+h/2)+pad_h)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    bike_roi = frame[y1:y2, x1:x2]
                    
                    # Unpack 4 values
                    violations, screenshot_path, annotated_crop, plate_text = self.violation_engine.detect_violations_on_bike(
                        bike_roi, frame_count, f"{frame_count}_{track_id}", (x, y)
                    )
                    
                    if annotated_crop.size > 0:
                        crop_rgb = cv2.cvtColor(annotated_crop, cv2.COLOR_BGR2RGB)
                        self.debug_crops.append(crop_rgb)
                    
                    if violations:
                        # 1. ALWAYS draw the Visual Alert (Red Box) on screen
                        orig_x1, orig_y1 = int(x-w/2), int(y-h/2)
                        orig_x2, orig_y2 = int(x+w/2), int(y+h/2)
                        cv2.rectangle(annotated_frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 0, 255), 3)
                        cv2.putText(annotated_frame, "VIOLATION!", (orig_x1, orig_y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # --- FIX 3: LOGIC TO PREVENT DUPLICATES ---
                        # Only increase count if this Track ID has NOT been caught before
                        if track_id not in self.violated_track_ids:
                            self.violated_track_ids.add(track_id) # Mark as caught
                            
                            self.violation_stats["Total Violations"] += 1
                            for v in violations: self.violation_stats[v] += 1
                            
                            # Only save screenshot for the FIRST time we catch them
                            if screenshot_path:
                                img_evidence = cv2.imread(screenshot_path)
                                if img_evidence is not None:
                                    img_evidence = cv2.cvtColor(img_evidence, cv2.COLOR_BGR2RGB)
                                    self.recent_violations.append({
                                        'image': img_evidence,
                                        'plate': plate_text if plate_text else "Unknown",
                                        'violations': violations
                                    })
                        # ------------------------------------------

        main_lane_moving = {d: 0.0 for d in self.main_lanes}
        main_lane_stopped = {d: 0.0 for d in self.main_lanes}
        for direct, val in raw_moving_density.items():
            for main in self.main_lanes:
                if main in direct: main_lane_moving[main] += val
        for direct, val in raw_stopped_density.items():
            for main in self.main_lanes:
                if main in direct: main_lane_stopped[main] += val
       
        time_step = 0.033
        for lane in self.main_lanes:
            if main_lane_stopped[lane] > 0.5 and main_lane_stopped[lane] >= main_lane_moving[lane]:
                self.lane_wait_timers[lane] += time_step        
            else: self.lane_wait_timers[lane] = 0.0
        
        total_vehicles = sum(vehicle_counts.values())
        congestion_msg = None
        if main_lane_stopped:
            worst_lane = max(main_lane_stopped, key=main_lane_stopped.get)
            if main_lane_stopped[worst_lane] > 4.0:
                congestion_msg = f"⚠️ CONGESTION: {worst_lane} Lane Blocked"
                return annotated_frame, main_lane_moving, worst_lane, 60, congestion_msg, self.lane_wait_timers, vehicle_counts, total_vehicles, self.violation_stats, self.recent_violations, self.debug_crops

        if sum(main_lane_moving.values()) > 0: active_lane = max(main_lane_moving, key=main_lane_moving.get)
        else: active_lane = max(self.lane_wait_timers, key=self.lane_wait_timers.get)
            
        count = main_lane_moving[active_lane]
        green_time = min(120, max(10, 5 + (count * 2)))
        
        return annotated_frame, main_lane_moving, active_lane, green_time, None, self.lane_wait_timers, vehicle_counts, total_vehicles, self.violation_stats, self.recent_violations, self.debug_crops