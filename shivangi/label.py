from ultralytics import YOLO
import os

teacher_model = YOLO('yolov8x.pt') 


IMAGE_DIR = 'all_extracted_frames/'
LABEL_DIR = 'all_extracted_labels/'
os.makedirs(LABEL_DIR, exist_ok=True)


target_classes = [2, 3, 5, 7]


mapping = {2: 0, 3: 1, 5: 2, 7: 3}

def auto_label():
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"Starting Auto-labeling for {len(images)} images...")

    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        
        results = teacher_model(img_path, conf=0.45, verbose=False)
        
        label_file = os.path.join(LABEL_DIR, img_name.rsplit('.', 1)[0] + '.txt')
        
        with open(label_file, 'w') as f:
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    
                    if cls_id in target_classes:
                        # (x_center, y_center, width, height)
                        xywh = box.xywhn[0].tolist()
                        new_cls = mapping[cls_id]
                        
                        f.write(f"{new_cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")
    
    print(f"Success! {len(images)} label files generated in '{LABEL_DIR}'.")

if __name__ == "__main__":
    auto_label()