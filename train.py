'''
Training pipeline for deepfake detection model
'''

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from datetime import datetime

from config import Config
from models import get_model


class DeepfakeTrainer:
    '''Handles model training, validation, and testing'''

    def __init__(self, config, model_name='efficientnet'):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.history = None

    # -------------------------------------------------------------------------
    # 📁 DATA LOADING (Memory Efficient)
    # -------------------------------------------------------------------------
    def load_full_dataset(self):
        '''Load dataset using ImageDataGenerator (memory efficient)'''
        print("\n[STEP 1/6] Loading Dataset (using generators)...")

        datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_gen = datagen.flow_from_directory(
            directory=os.path.join(self.config.DATASET_PATH, "train"),
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode="binary",
            shuffle=True
        )

        val_gen = datagen.flow_from_directory(
            directory=os.path.join(self.config.DATASET_PATH, "validate"),
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode="binary",
            shuffle=False
        )

        test_gen = datagen.flow_from_directory(
            directory=os.path.join(self.config.DATASET_PATH, "test"),
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode="binary",
            shuffle=False
        )

        print("\nDataset Summary:")
        print(f"  Train samples: {train_gen.samples}")
        print(f"  Validation samples: {val_gen.samples}")
        print(f"  Test samples: {test_gen.samples}")

        return train_gen, val_gen, test_gen

    # -------------------------------------------------------------------------
    # ⚙️ MODEL BUILDING AND TRAINING
    # -------------------------------------------------------------------------
    def create_data_augmentation(self):
        '''Data augmentation pipeline (optional)'''
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )

    def compile_model(self, model):
        '''Compile model'''
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        return model

    def get_callbacks(self, log_dir=None):
        '''Training callbacks'''
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(self.config.LOGS_PATH, f"{self.model_name}_{timestamp}")

        return [
            ModelCheckpoint(
                filepath=os.path.join(self.config.MODEL_SAVE_PATH, f'best_{self.model_name}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
        ]

    def train(self, train_gen, y_train, val_gen, y_val, use_augmentation=False):
        '''Train model using generators'''
        print(f"\nBuilding {self.model_name} model...")
        self.model = get_model(self.model_name, self.config.IMG_SIZE)
        self.model = self.compile_model(self.model)

        print("\nModel Architecture:")
        self.model.summary()

        print(f"\nStarting training for {self.config.EPOCHS} epochs...")
        self.history = self.model.fit(
            train_gen,
            epochs=self.config.EPOCHS,
            validation_data=val_gen,
            callbacks=self.get_callbacks(),
            verbose=1
        )

        print("\nTraining completed!")
        return self.history

    # -------------------------------------------------------------------------
    # 📊 EVALUATION AND SAVING
    # -------------------------------------------------------------------------
    def evaluate_on_test(self, test_gen, _unused=None):
        '''Evaluate model on test set'''
        print("\n[STEP 6/6] Evaluating on TEST SET...")
        results = self.model.evaluate(test_gen, verbose=1)
        print(f"\n✅ Test Results: {dict(zip(self.model.metrics_names, results))}")
        return results

    def save_model(self, filename=None):
        '''Save model and training history'''
        if filename is None:
            filename = f'{self.model_name}_final.h5'

        filepath = os.path.join(self.config.MODEL_SAVE_PATH, filename)
        self.model.save(filepath)
        print(f"\nModel saved to: {filepath}")

        # Save training history
        history_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{self.model_name}_history.json')
        with open(history_path, 'w') as f:
            history_dict = {k: [float(v) for v in val] for k, val in self.history.history.items()}
            json.dump(history_dict, f, indent=4)
        print(f"Training history saved to: {history_path}")

    def load_model(self, filepath):
        '''Load model from disk'''
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
        return self.model


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
