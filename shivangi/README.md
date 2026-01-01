Traffic AI: Video Preprocessing & Auto-Labeling Pipeline

This repository contains a streamlined pipeline to convert raw traffic surveillance footage into a structured dataset for training YOLO models.
It automates the manual labeling process by using a high-accuracy Teacher model (YOLOv8x) to annotate extracted frames.

Overview

The workflow is divided into two main stages:

Frame Extraction
Convert videos from multiple locations into a unified image dataset.

Auto-Labeling
Use YOLOv8x to detect vehicles and generate normalized bounding box coordinates in YOLO format.

Directory Structure

atcc_dataset/ Input: Raw video files organized by location
Talaimari/
Vodra/

all_extracted_frames/ Output: Every frame extracted from videos (generated)
all_extracted_labels/ Output: YOLO .txt files for each frame (generated)

preprocess.py Script 1: Video to Image conversion
label.py Script 2: Automated annotation using YOLOv8x

Getting Started

Prerequisites

Ensure Python is installed, then install the required dependencies:

pip install opencv-python ultralytics

Step 1: Extract Frames

Run the preprocessing script to scan the atcc_dataset folder and extract frames.
The script uses global indexing to ensure no filename overlaps across different locations.

python preprocess.py

Step 2: Generate Labels

Run the labeling script to automatically annotate the extracted frames.
The script uses the YOLOv8x (yolov8x.pt) model to detect traffic-related classes and remap them to custom class IDs.

python label.py

Configuration Details

Class Mapping

The pipeline filters standard COCO classes and remaps them to a simplified set for traffic analysis:

Original COCO ID Vehicle Type New Target ID
2 Car 0
3 Motorcycle 1
5 Bus 2
7 Truck 3

Detection Logic

Model: YOLOv8x (Extra Large) for high annotation accuracy
Confidence Threshold: 0.45
Output Format: Standard YOLO annotation files

Each label file follows the format:

<class_id> <x_center> <y_center> <width> <height>

All bounding box coordinates are normalized relative to image dimensions.


Dataset Expansion

To improve dataset diversity and model generalization, additional images and corresponding annotations were incorporated from an external public dataset.

External Dataset Source:
https://data.mendeley.com/datasets/nfc34n8svj/2

The external data was merged into the same directory structure and converted (where necessary) to match the YOLO annotation format and class mapping used in this project, ensuring consistency across the combined dataset.


Output

all_extracted_frames/ contains the extracted image frames
all_extracted_labels/ contains the auto-generated YOLO label files along with annotations from the external dataset

The final combined dataset can be directly used for training YOLOv5 or YOLOv8 models.

The finished preprocessed dataset is available on the drive link:
https://drive.google.com/drive/u/0/folders/1C0sEm7qHKzaUUqIluGmfDxBnKhBTqj3I