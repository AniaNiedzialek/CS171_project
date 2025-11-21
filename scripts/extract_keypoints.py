import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import pathlib
import argparse
import json

# Directory paths
INPUT_DIR = 'data/frames'
OUTPUT_DIR = 'data/keypoints'

# MediaPipe Pose has 33 landmarks 
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", 
    "right_eye_inner", "right_eye", "right_eye_outer", 
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

def extract_keypoints_mediapipe(pose_model, img_path):
    """
    Extract pose keypoints from a single image using MediaPipe.
    
    Args:
        pose_model: MediaPipe Pose model instance
        img_path: Path to the image file
        
    Returns:
        numpy array of shape (33, 4) containing [x, y, z, visibility] for each landmark
        Returns None if no pose is detected
    """
    image = cv2.imread(img_path)
    if image is None:
        return None
        
    # MediaPipe requires RGB format (OpenCV loads as BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose_model.process(image_rgb)
    
    if not results.pose_landmarks:
        return None
    
    # Extract all 33 landmarks with normalized coordinates [0.0, 1.0]
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        
    return np.array(landmarks, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Extract pose keypoints from frames using MediaPipe.")
    parser.add_argument("--static_mode", action="store_true", help="Set to true if frames are unrelated (slower but maybe more accurate per frame)")
    args = parser.parse_args()

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    
    # Use static_image_mode=True for independent frame processing 
    # model_complexity=2 gives highest accuracy (0=lite, 1=full, 2=heavy)
    
    print("Initializing MediaPipe Pose...")
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    # Find all frame directories (structure: data/frames/<category>/<video_name>/)
    frame_dirs = glob.glob(os.path.join(INPUT_DIR, "*", "*"))
    print(f"Found {len(frame_dirs)} video directories.")

    # Process each video directory (tqdm creates the loading bar)
    for frame_dir in tqdm(frame_dirs):
        frame_dir_path = pathlib.Path(frame_dir)
        category = frame_dir_path.parent.name
        video_name = frame_dir_path.name
        
        # Create output directory
        out_path = pathlib.Path(OUTPUT_DIR) / category / video_name
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Get all frames in chronological order
        img_paths = sorted(list(frame_dir_path.glob("*.jpg")))
        
        if not img_paths:
            continue
            
        # Store keypoints in two formats
        all_keypoints_np = []    # For ML/clustering (numpy array)
        all_keypoints_json = []  # For debugging/inspection (readable JSON)
        
        for i, img_path in enumerate(img_paths):
            kp = extract_keypoints_mediapipe(pose, str(img_path))
            
            frame_data = {
                "frame_index": i,
                "filename": img_path.name,
                "keypoints": {}
            }
            
            if kp is None:
                # No pose detected - use zero padding to maintain consistent array shape
                kp = np.zeros((33, 4), dtype=np.float32)
                frame_data["detected"] = False
            else:
                frame_data["detected"] = True
                # Build JSON with named body parts for each landmark
                for idx, name in enumerate(LANDMARK_NAMES):
                    frame_data["keypoints"][name] = {
                        "x": float(kp[idx, 0]),
                        "y": float(kp[idx, 1]),
                        "z": float(kp[idx, 2]),
                        "visibility": float(kp[idx, 3])
                    }
            
            all_keypoints_np.append(kp)
            all_keypoints_json.append(frame_data)
        
        np_save_path = out_path / "keypoints.npy"
        np.save(np_save_path, np.array(all_keypoints_np))
        
        json_save_path = out_path / "keypoints.json"
        with open(json_save_path, 'w') as f:
            json.dump(all_keypoints_json, f, indent=2)

    pose.close()
    print("Extraction complete.")

if __name__ == "__main__":
    main()
