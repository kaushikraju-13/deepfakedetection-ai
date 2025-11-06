'''
Neural network architectures for deepfake detection
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, Xception


def build_efficientnet_model(img_size=224, fine_tune_layers=20):
    '''
    Build deepfake detector using EfficientNetB0 with transfer learning
    
    Args:
        img_size: Input image size
        fine_tune_layers: Number of layers to fine-tune from the end
    
    Returns:
        Compiled Keras model
    '''
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Base model (EfficientNetB0 pre-trained on ImageNet)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling='avg'
    )
    
    # Fine-tune last layers
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    
    # Custom classification head
    x = base_model.output
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='EfficientNet_Deepfake_Detector')
    
    return model


def build_xception_model(img_size=224, fine_tune_layers=30):
    '''
    Build deepfake detector using XceptionNet with transfer learning
    XceptionNet is particularly good at detecting deepfakes
    
    Args:
        img_size: Input image size
        fine_tune_layers: Number of layers to fine-tune from the end
    
    Returns:
        Compiled Keras model
    '''
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Base model (Xception pre-trained on ImageNet)
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling='avg'
    )
    
    # Fine-tune last layers
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    
    # Custom classification head
    x = base_model.output
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Xception_Deepfake_Detector')
    
    return model


def build_custom_cnn(img_size=224):
    '''
    Custom CNN architecture built from scratch
    Lighter model, faster training, but potentially lower accuracy
    
    Args:
        img_size: Input image size
    
    Returns:
        Compiled Keras model
    '''
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name='Custom_CNN_Deepfake_Detector')
    
    return model


def get_model(model_name='efficientnet', img_size=224):
    '''
    Factory function to get desired model
    
    Args:
        model_name: 'efficientnet', 'xception', or 'custom_cnn'
        img_size: Input image size
    
    Returns:
        Keras model
    '''
    model_name = model_name.lower()
    
    if model_name == 'efficientnet':
        return build_efficientnet_model(img_size)
    elif model_name == 'xception':
        return build_xception_model(img_size)
    elif model_name == 'custom_cnn':
        return build_custom_cnn(img_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'efficientnet', 'xception', or 'custom_cnn'")


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    
    model = build_efficientnet_model(224)
    print(f"\nEfficientNet Model: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    
    model = build_xception_model(224)
    print(f"\nXception Model: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    
    model = build_custom_cnn(224)
    print(f"\nCustom CNN Model: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
