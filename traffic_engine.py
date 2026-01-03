import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

class TrafficLightManager:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Track history
        self.track_history = defaultdict(lambda: deque(maxlen=40))
        self.vehicle_directions = {} 
        self.verified_moving = set()
        
        # We focus on the 4 main traffic light phases
        self.main_lanes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.lane_wait_timers = {d: 0.0 for d in self.main_lanes}
        
        # Weights
        self.weights = {
            'Bus': 2.5, 'Truck': 2.5, 'Mini-bus': 2.0, 
            'Car': 1.0, 'SUV': 1.0, 'Sedan': 1.0, 'Hatchback': 1.0, 
            'Van': 1.0, 'MUV': 1.0, 'Tempo-traveller': 1.5,
            'Three-wheeler': 0.8, 'Two-wheeler': 0.5, 'Bicycle': 0.5
        }
        
        self.movement_threshold = 5 
        self.conf_threshold = 0.3

    def get_detailed_direction(self, track_id, current_x, current_y):
        if track_id not in self.track_history or len(self.track_history[track_id]) < 10:
            return None
        
        prev_x, prev_y = self.track_history[track_id][0]
        dx = current_x - prev_x
        dy = current_y - prev_y
        mag_x, mag_y = abs(dx), abs(dy)
        
        if mag_x < self.movement_threshold and mag_y < self.movement_threshold:
            return None 

        is_diagonal = (mag_x > self.movement_threshold) and (mag_y > self.movement_threshold)
        if is_diagonal:
            vertical = 'DOWN' if dy > 0 else 'UP'
            horizontal = 'RIGHT' if dx > 0 else 'LEFT'
            return f"{vertical}-{horizontal}"
        
        if mag_y > mag_x:
            return 'DOWN' if dy > 0 else 'UP'
        else:
            return 'RIGHT' if dx > 0 else 'LEFT'

    def is_vehicle_stopped(self, track_id):
        history = self.track_history[track_id]
        if len(history) < 20: return False
        start_x, start_y = history[0]
        end_x, end_y = history[-1]
        return np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) < 10 # Stricter stop check

    def process_frame(self, frame):
        results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_threshold, tracker="botsort.yaml")
        
        # Raw densities (including diagonals like UP-LEFT)
        raw_moving_density = defaultdict(float)
        raw_stopped_density = defaultdict(float)
        
        annotated_frame = frame.copy()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            names = results[0].names

            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w, h = box
                class_name = names[cls]
                weight = self.weights.get(class_name, 1.0)
                
                self.track_history[track_id].append((float(x), float(y)))
                
                # Verify active movement
                start_x, start_y = self.track_history[track_id][0]
                if np.sqrt((x - start_x)**2 + (y - start_y)**2) > 30:
                    self.verified_moving.add(track_id)
                
                raw_direction = self.get_detailed_direction(track_id, x, y)
                stopped = self.is_vehicle_stopped(track_id)
                
                if raw_direction:
                    self.vehicle_directions[track_id] = raw_direction

                final_direction = None
                if raw_direction:
                    final_direction = raw_direction
                elif track_id in self.vehicle_directions and track_id in self.verified_moving:
                    final_direction = self.vehicle_directions[track_id]
                
                if final_direction:
                    # Draw visual label (Keep diagonals for visual accuracy)
                    label = final_direction
                    color = (0, 255, 0)
                    
                    if stopped:
                        raw_stopped_density[final_direction] += weight
                        color = (0, 0, 255)
                        label = f"WAIT {final_direction}"
                    else:
                        raw_moving_density[final_direction] += weight
                        
                    cv2.putText(annotated_frame, label, (int(x), int(y)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
    
        # Map diagonals (DOWN-RIGHT) to main lanes (DOWN and RIGHT) for signal logic
        main_lane_moving = {d: 0.0 for d in self.main_lanes}
        main_lane_stopped = {d: 0.0 for d in self.main_lanes}

        # Helper to distribute counts
        def distribute_counts(source_dict, target_dict):
            for direct, val in source_dict.items():
                for main in self.main_lanes:
                    if main in direct: # e.g. 'DOWN' is in 'DOWN-RIGHT'
                        target_dict[main] += val

        distribute_counts(raw_moving_density, main_lane_moving)
        distribute_counts(raw_stopped_density, main_lane_stopped)

       
        time_step = 0.033 # approx per frame
        
        for lane in self.main_lanes:
            stopped_val = main_lane_stopped[lane]
            moving_val = main_lane_moving[lane]
            
            
            if stopped_val > 0.5 and stopped_val >= moving_val:
                self.lane_wait_timers[lane] += time_step        
            else:
               
                self.lane_wait_timers[lane] = 0.0
        
        # --- CONGESTION Part ---
        congestion_msg = None
        if main_lane_stopped:
            # Find worst blocked lane
            worst_lane = max(main_lane_stopped, key=main_lane_stopped.get)
            if main_lane_stopped[worst_lane] > 4.0: # Alert if >4 cars stopped
                congestion_msg = f"⚠️ CONGESTION: {worst_lane} Lane Blocked ({int(main_lane_stopped[worst_lane])} vehicles)"
                return annotated_frame, main_lane_moving, worst_lane, 60, congestion_msg, self.lane_wait_timers

        # --- SIGNAL TIMING Part---
        # Determine green light based on moving traffic
        if sum(main_lane_moving.values()) > 0:
            active_lane = max(main_lane_moving, key=main_lane_moving.get)
        else:
            # Fallback: if everyone is stopped, give green to the one waiting longest
            active_lane = max(self.lane_wait_timers, key=self.lane_wait_timers.get)
            
        count = main_lane_moving[active_lane]
        green_time = min(120, max(10, 5 + (count * 2)))
        
        return annotated_frame, main_lane_moving, active_lane, green_time, None, self.lane_wait_timers