'''
Utility functions for the deepfake detection system
'''

import os
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path


def print_banner(text, width=80):
    '''Print a formatted banner'''
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def save_json(data, filepath):
    '''Save data to JSON file'''
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to: {filepath}")


def load_json(filepath):
    '''Load data from JSON file'''
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def create_timestamp():
    '''Create timestamp string'''
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_files(directory, extensions):
    '''Count files with specific extensions in directory'''
    count = 0
    for ext in extensions:
        count += len(list(Path(directory).glob(f'*.{ext}')))
    return count


def get_video_info(video_path):
    '''Get video information'''
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    info = {
        'path': video_path,
        'total_frames': total_frames,
        'fps': int(fps),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_seconds': total_frames / fps if fps > 0 else 0
    }
    
    cap.release()
    return info


def extract_frame_at_time(video_path, time_seconds):
    '''Extract a single frame at specific time'''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(time_seconds * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None


def create_project_structure():
    '''Create complete project directory structure'''
    directories = [
        'data/real',
        'data/fake',
        'models',
        'results',
        'logs',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project structure created successfully!")
    print("\nDirectory structure:")
    for directory in directories:
        print(f"  ✓ {directory}/")


def download_sample_dataset_info():
    '''Instructions for downloading datasets'''
    print_banner("DATASET DOWNLOAD INSTRUCTIONS")
    
    print("Recommended Datasets for Deepfake Detection:\n")
    
    print("1. FaceForensics++")
    print("   - Website: https://github.com/ondyari/FaceForensics")
    print("   - Contains: Real and fake videos")
    print("   - Size: ~500GB (full), ~10GB (compressed)")
    print()
    
    print("2. Deepfake Detection Challenge (DFDC)")
    print("   - Kaggle: https://www.kaggle.com/c/deepfake-detection-challenge")
    print("   - Command: kaggle competitions download -c deepfake-detection-challenge")
    print("   - Size: ~470GB")
    print()
    
    print("3. Celeb-DF")
    print("   - Website: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("   - Contains: Celebrity deepfakes")
    print("   - Size: ~5.8GB")
    print()
    
    print("4. UADFV (For beginners - smaller dataset)")
    print("   - 49 real videos and 49 fake videos")
    print("   - Good for initial testing")
    print()
    
    print("Note: Make sure to organize your dataset as:")
    print("  data/")
    print("    ├── real/")
    print("    └── fake/")


if __name__ == "__main__":
    print("Utils module loaded successfully!")
