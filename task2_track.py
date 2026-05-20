import cv2
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from pathlib import Path

def setup_kalman_filter():
    # State vector: [x_center, y_center, x_velocity, y_velocity]
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # State transition matrix (assuming constant velocity: x_new = x + dx)
    kf.F = np.array([[1., 0., 1., 0.],
                     [0., 1., 0., 1.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
    
    # Measurement function (we only get x and y from our YOLO detector)
    kf.H = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.]])
    
    kf.P *= 1000.  # Initial uncertainty
    kf.R = np.array([[50., 0.],   # Measurement noise
                     [0., 50.]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.1, block_size=2) # Process noise
    
    return kf

def process_tracking():
    # Load the detections we saved in Task 1
    detections_df = pd.read_csv("drone_detections.csv")
    frames_dir = Path("frames")
    
    # Process each video
    for video_name in detections_df['video'].unique():
        print(f"Tracking drone in {video_name}...")
        
        # Get all detections for this specific video, sorted by frame order
        video_dets = detections_df[detections_df['video'] == video_name].sort_values('frame_name')
        
        kf = setup_kalman_filter()
        trajectory = []
        tracking_active = False
        missed_frames = 0
        max_missed = 5 # How many consecutive frames the filter will "guess" before giving up
        last_w, last_h = 50, 50 # Default bounding box size
        
        # Setup OpenCV Video Writer
        first_frame_path = str(frames_dir / video_name / video_dets.iloc[0]['frame_name'])
        sample_frame = cv2.imread(first_frame_path)
        h, w, _ = sample_frame.shape
        out_video_name = f"tracked_{video_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_name, fourcc, 5.0, (w, h)) # 5 FPS to match extraction
        
        # Loop through all extracted frames for this video
        all_frames = sorted((frames_dir / video_name).glob("*.jpg"))
        
        for frame_path in all_frames:
            frame_name = frame_path.name
            
            # Check if YOLO detected a drone in this specific frame
            det_row = video_dets[video_dets['frame_name'] == frame_name]
            has_detection = not det_row.empty
            
            if has_detection:
                z_x = det_row.iloc[0]['x_center']
                z_y = det_row.iloc[0]['y_center']
                last_w = det_row.iloc[0]['width']
                last_h = det_row.iloc[0]['height']
                z = np.array([[z_x], [z_y]]) # The measurement
                
                if not tracking_active:
                    # Start tracking! Set initial position to the first detection
                    kf.x = np.array([[z_x], [z_y], [0.], [0.]])
                    tracking_active = True
                    
                kf.predict()  # Step 1: Predict next state
                kf.update(z)  # Step 2: Update with actual measurement
                missed_frames = 0
                
            elif tracking_active:
                # YOLO missed the drone, but we are actively tracking it!
                kf.predict() # Step 1: Predict next state (no update step!)
                missed_frames += 1
                
                if missed_frames > max_missed:
                    tracking_active = False # We lost the drone for too long
            
            # The assignment: "Output video must contain ONLY the frames where the drone is present"
            if tracking_active:
                img = cv2.imread(str(frame_path))
                
                # Get the Kalman Filter's current best estimate of the center
                est_x = int(kf.x[0, 0])
                est_y = int(kf.x[1, 0])
                trajectory.append((est_x, est_y))
                
                # Draw the 2D trajectory as a yellow line
                if len(trajectory) > 1:
                    cv2.polylines(img, [np.array(trajectory)], isClosed=False, color=(0, 255, 255), thickness=3)
                    
                # Draw the bounding box
                top_left = (int(est_x - last_w/2), int(est_y - last_h/2))
                bottom_right = (int(est_x + last_w/2), int(est_y + last_h/2))
                
                # Green box = Confirmed Detection. Red box = Kalman Filter Predicting a missed frame
                color = (0, 255, 0) if has_detection else (0, 0, 255)
                cv2.rectangle(img, top_left, bottom_right, color, 3)
                
                out.write(img)
        
        out.release()
        print(f"Finished generating {out_video_name}")

if __name__ == "__main__":
    process_tracking()