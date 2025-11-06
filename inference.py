'''
Inference module for making predictions on new data
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras

from preprocessing import VideoProcessor


class DeepfakeDetector:
    '''Production-ready deepfake detector for inference'''
    
    def __init__(self, model_path, img_size=224):
        '''
        Initialize the detector
        
        Args:
            model_path: Path to trained model file
            img_size: Input image size
        '''
        self.model = keras.models.load_model(model_path)
        self.processor = VideoProcessor(img_size)
        self.img_size = img_size
        print(f"Model loaded successfully from: {model_path}")
    
    def predict_image(self, image_path, threshold=0.5):
        '''
        Predict if an image is fake or real
        
        Args:
            image_path: Path to image file
            threshold: Classification threshold (default 0.5)
        
        Returns:
            Dictionary with prediction results
        '''
        # Process image
        face = self.processor.process_image(image_path)
        
        if face is None:
            return {
                'success': False,
                'error': 'No face detected in image',
                'label': None,
                'confidence': None
            }
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        # Predict
        prediction = self.model.predict(face, verbose=0)[0][0]
        
        # Determine label
        label = "FAKE" if prediction > threshold else "REAL"
        confidence = prediction if prediction > threshold else 1 - prediction
        
        return {
            'success': True,
            'label': label,
            'confidence': float(confidence),
            'probability_fake': float(prediction),
            'probability_real': float(1 - prediction),
            'threshold': threshold
        }
    
    def predict_video(self, video_path, num_frames=10, threshold=0.5, 
                     aggregation='mean'):
        '''
        Predict if a video is fake or real
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            threshold: Classification threshold
            aggregation: How to aggregate predictions ('mean', 'max', 'voting')
        
        Returns:
            Dictionary with prediction results
        '''
        # Process video
        faces = self.processor.process_video(video_path, num_frames)
        
        if len(faces) == 0:
            return {
                'success': False,
                'error': 'No faces detected in video',
                'label': None,
                'confidence': None
            }
        
        # Predict on all frames
        predictions = self.model.predict(faces, verbose=0).flatten()
        
        # Aggregate predictions
        if aggregation == 'mean':
            final_prediction = np.mean(predictions)
        elif aggregation == 'max':
            final_prediction = np.max(predictions)
        elif aggregation == 'voting':
            votes = (predictions > threshold).astype(int)
            final_prediction = 1.0 if np.mean(votes) > 0.5 else 0.0
        else:
            final_prediction = np.mean(predictions)
        
        # Determine label
        label = "FAKE" if final_prediction > threshold else "REAL"
        confidence = final_prediction if final_prediction > threshold else 1 - final_prediction
        
        return {
            'success': True,
            'label': label,
            'confidence': float(confidence),
            'probability_fake': float(final_prediction),
            'probability_real': float(1 - final_prediction),
            'num_frames_analyzed': len(faces),
            'frame_predictions': predictions.tolist(),
            'aggregation_method': aggregation,
            'threshold': threshold
        }
    
    def predict_batch(self, file_paths, threshold=0.5):
        '''
        Predict on multiple files
        
        Args:
            file_paths: List of file paths (images or videos)
            threshold: Classification threshold
        
        Returns:
            List of prediction results
        '''
        results = []
        
        for file_path in file_paths:
            # Determine file type
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                result = self.predict_video(file_path, threshold=threshold)
            else:
                result = self.predict_image(file_path, threshold=threshold)
            
            result['file_path'] = file_path
            results.append(result)
        
        return results
    
    def predict_from_array(self, image_array, threshold=0.5):
        '''
        Predict from numpy array (useful for web applications)
        
        Args:
            image_array: Numpy array of image (BGR format)
            threshold: Classification threshold
        
        Returns:
            Dictionary with prediction results
        '''
        # Detect face
        face = self.processor.detect_face(image_array)
        
        if face is None:
            return {
                'success': False,
                'error': 'No face detected',
                'label': None,
                'confidence': None
            }
        
        # Preprocess
        processed = self.processor.preprocess_frame(face)
        processed = np.expand_dims(processed, axis=0)
        
        # Predict
        prediction = self.model.predict(processed, verbose=0)[0][0]
        
        label = "FAKE" if prediction > threshold else "REAL"
        confidence = prediction if prediction > threshold else 1 - prediction
        
        return {
            'success': True,
            'label': label,
            'confidence': float(confidence),
            'probability_fake': float(prediction),
            'probability_real': float(1 - prediction),
            'threshold': threshold
        }


if __name__ == "__main__":
    print("Inference module loaded successfully!")
    print("\nExample usage:")
    print("detector = DeepfakeDetector('models/best_model.h5')")
    print("result = detector.predict_image('test_image.jpg')")
    print("print(result)")
