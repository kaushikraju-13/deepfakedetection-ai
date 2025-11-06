'''
Configuration file for Deepfake Detection System
Authors: Nischay Upadhya P, Supreeth Gutti, Kaushik Raju S, Nandeesha B
'''

import os

class Config:
    '''Configuration parameters for the deepfake detection system'''
    
    # Project paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, 'data', 'dataset')
    TRAIN_REAL_PATH = os.path.join(DATASET_PATH, 'train', 'real')
    TRAIN_FAKE_PATH = os.path.join(DATASET_PATH, 'train', 'fake')
    TEST_REAL_PATH = os.path.join(DATASET_PATH, 'test', 'real')
    TEST_FAKE_PATH = os.path.join(DATASET_PATH, 'test', 'fake')
    VAL_REAL_PATH = os.path.join(DATASET_PATH, 'validate', 'real')
    VAL_FAKE_PATH = os.path.join(DATASET_PATH, 'validate', 'fake')

    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
    RESULTS_PATH = os.path.join(BASE_DIR, 'results')
    LOGS_PATH = os.path.join(BASE_DIR, 'logs')

    IMG_SIZE = 224
    CHANNELS = 3
    FRAMES_PER_VIDEO = 10
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_SEED = 42
    FACE_DETECTION_CONFIDENCE = 0.9
    FACE_PADDING = 20

    @staticmethod
    def create_directories():
        for path in [
            Config.MODEL_SAVE_PATH,
            Config.RESULTS_PATH,
            Config.LOGS_PATH,
        ]:
            os.makedirs(path, exist_ok=True)
        print("All necessary directories verified!")