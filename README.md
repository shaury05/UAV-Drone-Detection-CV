# UAV Drone Detection and Tracking

## Overview
This project detects and tracks UAV drones in video footage using a deep learning object detector (YOLOv8) and a Kalman filter for multi-object tracking (MOT). 

## Links
* **Hugging Face Dataset (Parquet)**: [Shauryaaa05/uav-drone-detections](https://huggingface.co/datasets/Shauryaaa05/uav-drone-detections)
* **Tracked Video 1**: [Watch on YouTube](https://youtu.be/iDUc0NynjV8)
* **Tracked Video 2**: [Watch on YouTube](https://youtu.be/TnZkiKwXTbY)

## Task 1: Dataset Choice and Detector Configuration
While researching datasets, I evaluated the **VisDrone** dataset and **Roboflow Universe Drone Detection** datasets, which provide excellent bounding box labels for aerial targets. However, because the assignment states that fine-tuning is encouraged but not required, I opted to configure a pre-trained **Ultralytics YOLOv8s** model. Because standard COCO does not have a "drone" class, YOLOv8 frequently classifies distant drones as birds (class 14), airplanes (class 16), or kites (class 17). The detection script filters the inference results for these specific classes to successfully identify the UAV. Frames containing detections were cropped and saved to generate the sample dataset.

## Task 2: Kalman Filter State Design and Noise Parameters
I implemented a Kalman filter using the `filterpy` library to track the drone's 2D trajectory. 
* **State Vector**: The state vector is 4-dimensional: `[x, y, dx, dy]`, representing the 2D pixel coordinates of the bounding box center (`x`, `y`) and their respective velocities (`dx`, `dy`).
* **State Transition**: Modeled using a constant velocity kinematic model.
* **Measurement Vector**: 2-dimensional `[x, y]` provided by the YOLOv8 detector.
* **Noise Parameters**: 
  * Initial uncertainty (`P`) was scaled by 1000.
  * Measurement noise (`R`) was set to 50 for both x and y to account for bounding box jitter.
  * Process noise (`Q`) was generated using `Q_discrete_white_noise` with a variance of 0.1, allowing the filter to adapt to smooth changes in velocity.

## Failure Cases and Handling Missed Detections
**Failure Cases**: The YOLO detector occasionally misses the drone, especially when it shrinks to a very small pixel size in the distance or blends into dark backgrounds like trees.

**Handling Missed Detections**: The Kalman filter is designed to handle temporary detection failures. If the YOLO detector fails to find the drone in a frame, the tracker skips the `update()` step and only runs the `predict()` step. This allows the tracker to "guess" the drone's location based on its last known velocity. 
* The tracker is configured with a `max_missed` threshold of 5 frames. 
* If the drone is missing for fewer than 5 frames, a red bounding box is drawn to indicate a predicted state. 
* If it exceeds 5 consecutive missed frames, the tracker deactivates until the drone is detected again.
