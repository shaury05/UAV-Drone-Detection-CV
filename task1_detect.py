import os
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

def process_frames():
    # Load the pre-trained YOLOv8 model (it will download automatically)
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8s.pt') 
    
    # Setup directories
    frames_dir = Path("frames")
    output_dir = Path("detections")
    output_dir.mkdir(exist_ok=True)
    
    # We will store detection data in a list to save as a CSV later (for Task 2)
    detection_data = []
    
    # We are looking for classes that YOLO commonly confuses drones with (0: person, 14: bird, 16: airplane, 17: kite)
    target_classes = [14, 16, 17] 

    print("Starting detection...")
    # Loop through all folders in the frames directory
    for video_folder in frames_dir.iterdir():
        if not video_folder.is_dir():
            continue
            
        print(f"Processing {video_folder.name}...")
        
        # Loop through all images in the video folder
        for img_path in sorted(video_folder.glob("*.jpg")):
            # Run YOLOv8 inference
            results = model(str(img_path), verbose=False)
            
            drone_detected = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class ID
                    cls_id = int(box.cls[0])
                    
                    # If the detected object is one of our target classes
                    if cls_id in target_classes:
                        drone_detected = True
                        
                        # Get bounding box coordinates (center x, center y, width, height)
                        x_center, y_center, w, h = box.xywh[0].tolist()
                        
                        # Save the detection info for our Kalman Filter later
                        detection_data.append({
                            "video": video_folder.name,
                            "frame_name": img_path.name,
                            "frame_path": str(img_path),
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": w,
                            "height": h,
                            "confidence": float(box.conf[0])
                        })
            
            # The assignment requires saving frames that contain at least one detection
            if drone_detected:
                # Read the image, draw the YOLO boxes, and save it
                img = cv2.imread(str(img_path))
                annotated_img = results[0].plot() # This draws the bounding boxes
                
                # Create a specific name like "video_1_frame_0001.jpg"
                save_name = f"{video_folder.name}_{img_path.name}"
                save_path = output_dir / save_name
                cv2.imwrite(str(save_path), annotated_img)

    # Save the coordinates to a CSV file so we can use them in Task 2
    df = pd.DataFrame(detection_data)
    df.to_csv("drone_detections.csv", index=False)
    print(f"Done! Saved {len(df)} detections to drone_detections.csv and images to the 'detections' folder.")

if __name__ == "__main__":
    process_frames()