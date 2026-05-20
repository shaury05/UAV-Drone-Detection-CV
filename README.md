# 🚁 Autonomous UAV Drone Detection & Predictive Tracking

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer_Vision-yellow.svg)](https://ultralytics.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Dataset_Hosted-orange.svg)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)

## 🚀 The Vision
Unstructured video footage is messy. This project is an end-to-end Computer Vision and Data Engineering pipeline built to ingest raw, noisy video streams and extract clean, actionable trajectory data. By combining deep learning object detection with probabilistic kinetic tracking, this system autonomously finds and tracks Unmanned Aerial Vehicles (UAVs)—even when they are tiny, distant, or temporarily hidden by clouds.

### 🎥 Live Demonstrations & Data
* **[View the Tracked Video 1](https://youtu.be/iDUc0NynjV8)** (YouTube)
* **[View the Tracked Video 2](https://youtu.be/TnZkiKwXTbY)** (YouTube)
* **[Explore the Extracted Dataset](https://huggingface.co/datasets/Shauryaaa05/uav-drone-detections)** (Hosted on Hugging Face in optimized Parquet format)

---

## 🛠️ Tech Stack
* **AI & Computer Vision:** Ultralytics YOLOv8, OpenCV, FilterPy (Kalman Filters)
* **Data Engineering & Processing:** Hugging Face `datasets`, FFMPEG, Pandas, Apache Parquet
* **Environment & DevOps:** Docker, Git

---

## 🧠 Engineering Architecture: How It Works

### 1. Robust Data Ingestion
The pipeline processes raw `.mp4` video files, isolating individual frames using `ffmpeg` within an entirely containerized **Docker** environment. This guarantees dependency consistency, meaning the pipeline will run flawlessly on any machine without local environment conflicts.

### 2. Deep Learning Detection (Finding the Needle in the Haystack)
The system leverages the **YOLOv8s** architecture. Because standard COCO weights lack a dedicated "drone" class, the inference script dynamically filters and remaps spatially similar classes (like birds, distant airplanes, and kites). This engineering decision allows the pre-trained model to successfully identify distant UAVs against complex, moving backgrounds without requiring a massive, computationally expensive fine-tuning phase.

### 3. Predictive Tracking (When the AI goes blind)
**The Problem:** Deep learning detectors fail. If a drone shrinks to a few pixels, enters a shadow, or flies behind a tree, YOLO drops the bounding box.
**The Solution:** I engineered a 4D **Kalman Filter** `[x, y, dx, dy]` utilizing a constant velocity kinematic model to track the drone's trajectory. 
* When the AI successfully detects the drone, the Kalman Filter *updates* its tracking logic (visualized as a **Green** bounding box).
* When the AI drops the frame (occlusion), the tracker seamlessly steps in to *predict* the drone's location based on its last known velocity vector (visualized as a **Red** bounding box). 
* A memory threshold prevents false-positive drift, terminating the track if the target is lost for more than 5 consecutive frames.

### 4. Cloud Deployment & Data Structuring
The final, verified detections are not just left on a local hard drive. The pipeline automatically aggregates the visual data, converts it into a highly efficient, query-ready **Apache Parquet** format, and programmatically pushes it to the **Hugging Face Hub**. 

---
*Designed and engineered by Shaury Pratap Singh.*
