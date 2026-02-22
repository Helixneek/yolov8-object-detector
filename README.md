# Yolov8 Realtime Object Detector
Custom Object Detection Pipeline with Synthetic Data Generation

## Overview
This repository contains a computer vision pipeline for training a custom object detection model. To address the problem of data scarcity, the project implements a synthetic data generation script to procedurally create and auto-annotate training images. The core detection model is based on the YOLOv8 architecture, fine-tuned via transfer learning for real-time inference on consumer-grade hardware.

## Pipeline

The system is divided into three modules:

1. **Synthetic Data Generation (OpenCV):** A procedural generation script that overlays target foreground objects onto randomized backgrounds. The script applies geometric transformations (scaling, rotation, translation) and photometric distortions (brightness, contrast) to ensure dataset variance. 
   
2. **Deterministic Auto-Annotation:**
   During the generation phase, bounding box coordinates are calculated mathematically based on the applied transformations. These coordinates are normalized and exported directly into the standard `.txt` format required by the YOLO architecture, eliminating the need for manual data labeling.

3. **Model Fine-Tuning (YOLOv8):**
   Transfer learning is applied to a pre-trained YOLOv8 weights file. The model is trained over the synthetic dataset with mixed-precision training enabled to optimize VRAM utilization.

## Technical Specifications
* **Languages & Libraries:** Python, PyTorch, OpenCV, Ultralytics (YOLOv8)
* **Hardware Profile:** Tested on my Nvidia GTX 1660 Ti (6GB VRAM)
* **Performance Metric:** Achieved 98% mAP (Mean Average Precision) on the validation set.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/custom-object-detector.git](https://github.com/yourusername/custom-object-detector.git)
   cd custom-object-detector

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## Usage

1. Generate the dataset
   Run the synthetic data generator to create the images and bounding box labels.
   ```bash
   python src/generate_synthetic_data.py --samples 1000

2. Train the model
   Start the transfer learning process.
   ```bash
   python src/train_yolo.py --epochs 50 --batch 16

3. Run inference
   Test the trained weights (best.pt) on a new image or video stream.
   ```bash
   python src/inference.py --source path/to/test/image.jpg
