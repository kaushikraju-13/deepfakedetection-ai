'''
Data preprocessing and face detection module
Handles video processing, frame extraction, and face detection
'''

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mtcnn import MTCNN
import warnings
warnings.filterwarnings('ignore')


class VideoProcessor:
    '''Handles video processing, frame extraction, and face detection'''
    
    def __init__(self, img_size=224, face_padding=20):
        self.img_size = img_size
        self.face_padding = face_padding
        self.face_detector = MTCNN()
    
    def extract_frames(self, video_path, num_frames=10):
        '''Extract evenly spaced frames from video'''
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return []
        
        # Calculate frame indices to extract
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def detect_face(self, frame):
        '''Detect and extract face from frame using MTCNN'''
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.face_detector.detect_faces(rgb_frame)
        
        if len(detections) > 0:
            # Get the largest face
            detection = max(detections, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = detection['box']
            
            # Add padding
            x = max(0, x - self.face_padding)
            y = max(0, y - self.face_padding)
            w = w + 2 * self.face_padding
            h = h + 2 * self.face_padding
            
            face = rgb_frame[y:y+h, x:x+w]
            return face
        return None
    
    def preprocess_frame(self, frame):
        '''Resize and normalize frame'''
        if frame is None:
            return None
        
        # Resize
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def process_video(self, video_path, num_frames=10):
        '''Complete pipeline: extract frames, detect faces, preprocess'''
        frames = self.extract_frames(video_path, num_frames)
        processed_faces = []
        
        for frame in frames:
            face = self.detect_face(frame)
            if face is not None:
                processed = self.preprocess_frame(face)
                if processed is not None:
                    processed_faces.append(processed)
        
        return np.array(processed_faces)
    
    def process_image(self, image_path):
        '''Process a single image'''
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        face = self.detect_face(img)
        if face is not None:
            return self.preprocess_frame(face)
        return None


class DatasetBuilder:
    '''Build dataset from videos/images'''
    
    def __init__(self, config):
        self.config = config
        self.processor = VideoProcessor(config.IMG_SIZE, config.FACE_PADDING)
    
    def load_from_videos(self, real_dir, fake_dir, num_frames=10, limit=None):
        '''Load dataset from video files'''
        X, y = [], []
        
        # Process real videos
        print("Processing REAL videos...")
        real_videos = list(Path(real_dir).glob('*.mp4')) + list(Path(real_dir).glob('*.avi'))
        if limit:
            real_videos = real_videos[:limit]
        
        for video_path in tqdm(real_videos, desc="Real videos"):
            faces = self.processor.process_video(str(video_path), num_frames)
            for face in faces:
                X.append(face)
                y.append(0)  # 0 for real
        
        # Process fake videos
        print("Processing FAKE videos...")
        fake_videos = list(Path(fake_dir).glob('*.mp4')) + list(Path(fake_dir).glob('*.avi'))
        if limit:
            fake_videos = fake_videos[:limit]
        
        for video_path in tqdm(fake_videos, desc="Fake videos"):
            faces = self.processor.process_video(str(video_path), num_frames)
            for face in faces:
                X.append(face)
                y.append(1)  # 1 for fake
        
        return np.array(X), np.array(y)
    
    def load_from_images(self, real_dir, fake_dir):
        '''Load dataset from image files'''
        X, y = [], []
        
        # Process real images
        print("Processing REAL images...")
        real_images = list(Path(real_dir).glob('*.jpg')) + list(Path(real_dir).glob('*.png'))
        for img_path in tqdm(real_images, desc="Real images"):
            face = self.processor.process_image(str(img_path))
            if face is not None:
                X.append(face)
                y.append(0)
        
        # Process fake images
        print("Processing FAKE images...")
        fake_images = list(Path(fake_dir).glob('*.jpg')) + list(Path(fake_dir).glob('*.png'))
        for img_path in tqdm(fake_images, desc="Fake images"):
            face = self.processor.process_image(str(img_path))
            if face is not None:
                X.append(face)
                y.append(1)
        
        return np.array(X), np.array(y)


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    builder = DatasetBuilder(config)
    print("DatasetBuilder initialized successfully!")
