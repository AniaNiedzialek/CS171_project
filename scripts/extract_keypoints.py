import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import pathlib
import argparse
import json
from typing import Dict, Any

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

def extract_keypoints(
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR,
    static_mode: bool = True,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    mp_pose = mp.solutions.pose
    if verbose:
        print("Initializing MediaPipe Pose...")
    pose = mp_pose.Pose(
        static_image_mode=static_mode,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
    )

    # Find all frame directories (structure: data/frames/<category>/<video_name>/)
    frame_dirs = glob.glob(os.path.join(input_dir, "*", "*"))
    if verbose:
        print(f"Found {len(frame_dirs)} video directories.")

    results: Dict[str, Dict[str, Any]] = {}

    for frame_dir in tqdm(frame_dirs):
        frame_dir_path = pathlib.Path(frame_dir)
        category = frame_dir_path.parent.name
        video_name = frame_dir_path.name

        out_path = pathlib.Path(output_dir) / category / video_name
        out_path.mkdir(parents=True, exist_ok=True)

        # Get all frames in chronological order
        img_paths = sorted(list(frame_dir_path.glob("*.jpg")))
        if not img_paths:
            continue

        # Store keypoints in two formats
        all_keypoints_np = []    # For ML/clustering (numpy array)
        all_keypoints_json = []  # For debugging/inspection (readable JSON)

        num_detected = 0
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
                num_detected += 1
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

        # Save
        np_save_path = out_path / "keypoints.npy"
        np.save(np_save_path, np.array(all_keypoints_np))

        json_save_path = out_path / "keypoints.json"
        with open(json_save_path, 'w') as f:
            json.dump(all_keypoints_json, f, indent=2)

        # record stats
        key = f"{category}/{video_name}"
        results[key] = {
            "num_frames": len(img_paths),
            "num_detected": num_detected,
            "npy_path": str(np_save_path),
            "json_path": str(json_save_path),
        }

    pose.close()
    if verbose:
        print("Extraction complete.")

    return results