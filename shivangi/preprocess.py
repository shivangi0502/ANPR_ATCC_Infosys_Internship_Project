import cv2
import os

# Configuration
input_folders = ['Talaimari', 'Vodra']
#images will go into this single folder
output_dir = 'all_extracted_frames'

def preprocess_videos():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    global_count = 1  

    for folder in input_folders:
    
        video_folder_path = os.path.join('atcc_dataset', folder)
        
        if not os.path.exists(video_folder_path):
            print(f"Skipping {folder}: Folder not found.")
            continue

        for video_name in os.listdir(video_folder_path):
            if video_name.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_folder_path, video_name)
                cap = cv2.VideoCapture(video_path)
                
                print(f"Processing: {video_name} from {folder}...")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    
                    img_filename = f"frame_{global_count:05d}.jpg"
                    cv2.imwrite(os.path.join(output_dir, img_filename), frame)
                    
                    global_count += 1
                
                cap.release()

    print(f"\nSuccess! Total frames extracted: {global_count - 1}")
    print(f"All images are located in: {output_dir}")

if __name__ == "__main__":
    preprocess_videos()