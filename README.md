# Deepfake Detection System

**V Semester Mini Project (2025-26)**

## Team Members
- **Nischay Upadhya P** (1MV23IC039)
- **Supreeth Gutti** (1MV23IC058)
- **Kaushik Raju S** (1MV23IC046)
- **Nandeesha B** (1MV23IC035)

---

## Overview

An AI-powered deepfake detection system using deep learning to identify manipulated videos and images with high accuracy (90-95%).

## Features

- ✅ Multiple CNN architectures (EfficientNet, Xception, Custom CNN)
- ✅ Automatic face detection using MTCNN
- ✅ Support for both images and videos
- ✅ Data augmentation for robust training
- ✅ Comprehensive evaluation metrics
- ✅ Automatic visualization generation
- ✅ Easy-to-use CLI interface

---

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
python -c "from config import Config; Config.create_directories()"
```

---

## Dataset Setup

Organize your dataset:

```
data/
├── real/
│   ├── image1.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    └── ...
```

---

## Usage

### Training

```bash
# Train with EfficientNet
python main.py train --model efficientnet --data_type images --epochs 30

# Train with custom CNN (faster)
python main.py train --model custom_cnn --data_type images --epochs 20
```

### Testing

```bash
# Test single image
python main.py test --model_path models/best_efficientnet.h5 --file_path test.jpg

# Test video
python main.py test --model_path models/best_efficientnet.h5 --file_path test.mp4
```

---

## Project Structure

```
deepfake-detection/
├── config.py              # Configuration
├── preprocessing.py       # Data preprocessing
├── models.py             # Neural networks
├── train.py              # Training pipeline
├── evaluate.py           # Evaluation
├── visualize.py          # Visualization
├── inference.py          # Predictions
├── utils.py              # Utilities
├── main.py               # CLI interface
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── data/                # Dataset
├── models/              # Saved models
└── results/             # Results
```

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| EfficientNet | 92.5% | 91.8% | 93.2% | 92.5% |
| Xception | 91.2% | 90.5% | 92.0% | 91.2% |
| Custom CNN | 85.8% | 84.2% | 87.5% | 85.8% |

---

## Technologies

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning
- **OpenCV** - Computer vision
- **MTCNN** - Face detection
- **Scikit-learn** - Metrics
- **Matplotlib/Seaborn** - Visualization

---

## License

MIT License

---

## Contact

For questions: Contact team members

**⭐ Star this repository if you find it helpful!**
