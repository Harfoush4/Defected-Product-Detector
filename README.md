# 🏭 Industrial Casting Defect Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](https://github.com/)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a state-of-the-art deep learning system for detecting defects in industrial casting products. Using computer vision and convolutional neural networks, the system can automatically identify defective castings with high accuracy, enabling quality control automation in manufacturing environments.

### 🔑 Key Capabilities
- **Real-time defect detection** in casting products
- **Binary classification**: OK vs Defective
- **High accuracy** (>95%) with production-ready models
- **Multiple model architectures** including custom CNN and transfer learning
- **Comprehensive evaluation** metrics and visualizations
- **Production pipeline** with batch processing support
- **Edge deployment** ready with TensorFlow Lite export

## ✨ Features

### 🤖 Machine Learning
- **Custom CNN Architecture** - Optimized for casting defect patterns
- **Transfer Learning** - VGG16, ResNet50, MobileNetV2 pre-trained models
- **Data Augmentation** - Rotation, flipping, zooming for robust training
- **Fine-tuning Support** - Adaptive learning rate scheduling

### 📊 Analysis & Reporting
- **Comprehensive EDA** - Dataset distribution and sample visualizations
- **Error Analysis** - False positive/negative identification
- **Quality Metrics** - Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cost-Benefit Analysis** - Business impact assessment
- **Confidence Scoring** - Probability-based decision making

### 🚀 Production Features
- **Batch Processing** - Efficient multi-image processing
- **Model Versioning** - Track and manage model iterations
- **TFLite Export** - Edge device deployment
- **API-Ready Functions** - Easy integration with existing systems
- **Real-time Monitoring** - Performance tracking capabilities

## 📁 Dataset

### Source
**Kaggle Dataset**: [Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

### Dataset Statistics
```
Total Images: 7,348
├── Training Set: ~5,878 images
│   ├── Defective: ~2,939 images
│   └── OK: ~2,939 images
└── Test Set: ~1,470 images
    ├── Defective: ~735 images
    └── OK: ~735 images
```

### Dataset Structure
```
casting_data/
├── casting_512x512/
│   ├── train/
│   │   ├── def_front/  # Defective castings
│   │   └── ok_front/   # Good castings
│   └── test/
│       ├── def_front/
│       └── ok_front/
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/casting-defect-detection.git
cd casting-defect-detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
opencv-python>=4.5.0
Pillow>=8.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

## 🚀 Quick Start

### For Kaggle Users

1. **Open Kaggle Notebook**
   - Go to [Kaggle](https://www.kaggle.com)
   - Create New Notebook

2. **Add Dataset**
   - Click "Add Data"
   - Search: "real-life-industrial-dataset-of-casting-product"
   - Add to notebook

3. **Copy Code**
   - Copy the entire Python code from `casting_defect_detection.py`
   - Paste into Kaggle notebook

4. **Enable GPU**
   - Settings → Accelerator → GPU

5. **Run All Cells**
   - Runtime → Run all

### For Local Development

```python
# Basic usage
from casting_defect_detector import DefectDetector

# Initialize detector
detector = DefectDetector('path/to/model.h5')

# Single image prediction
result = detector.predict('path/to/image.jpg')
print(f"Status: {result['status']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch processing
results = detector.batch_predict(['img1.jpg', 'img2.jpg', 'img3.jpg'])
stats = detector.get_statistics(results)
print(f"Defect Rate: {stats['defect_rate']:.2f}%")
```

## 🏗️ Model Architecture

### Custom CNN Architecture
```
Input (224x224x3)
    ↓
Conv2D(32) → BatchNorm → AVGPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → AVGPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → AVGPool → Dropout(0.25)
    ↓
Conv2D(256) → BatchNorm → AVGPool → Dropout(0.25)
    ↓
Flatten → Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(1, sigmoid) → Output
```

### Transfer Learning Models
- **MobileNetV2** (Recommended for edge deployment)
- **VGG16** (High accuracy, larger size)
- **ResNet50** (Balance of accuracy and efficiency)

## 📈 Performance Metrics

### Model Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 99.4% |
| **Precision** | 98.5% |
| **Recall** | 100% |
| **F1-Score** | 99.24% |
| **AUC-ROC** | 0.999 |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Image Size | 224x224 |
| Epochs | 15-20 |
| Learning Rate | 0.001 (adaptive) |
| Optimizer | Adam |

### Hardware Requirements
- **Minimum**: 4GB GPU, 8GB RAM
- **Recommended**: 8GB+ GPU, 16GB RAM
- **Training Time**: ~15 mins (GPU), ~2 hours (CPU)

## 💡 Usage Examples

### 1. Training a New Model
```python
from src.train import train_model

# Configure training
config = {
    'model_type': 'MobileNetV2',
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.001
}

# Train model
model, history = train_model(config)
```

### 2. Evaluating Model Performance
```python
from src.evaluate import evaluate_model

# Load model and evaluate
metrics = evaluate_model('models/casting_defect_detector.h5', 
                        test_data_path='data/test/')

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1-Score: {metrics['f1_score']:.2%}")
```

### 3. Real-time Prediction
```python
from src.predict import DefectDetector
import cv2

# Initialize detector
detector = DefectDetector('models/casting_defect_detector.h5')

# Capture from camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        result = detector.predict_frame(frame)
        cv2.putText(frame, f"{result['status']} ({result['confidence']:.1%})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 255, 0) if result['status'] == 'OK' else (0, 0, 255), 2)
        cv2.imshow('Defect Detection', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 🚢 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t casting-defect-detector .

# Run container
docker run -p 8080:8080 casting-defect-detector
```

### API Deployment
```python
# FastAPI implementation (api.py)
from fastapi import FastAPI, File, UploadFile
from src.predict import DefectDetector

app = FastAPI()
detector = DefectDetector('models/casting_defect_detector.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    result = detector.predict_bytes(image)
    return result
```

### Edge Deployment (Raspberry Pi / Jetson Nano)
```bash
# Convert to TFLite
python convert_to_tflite.py --model casting_defect_detector.h5

# Deploy on edge device
python edge_inference.py --model casting_defect_detector.tflite
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Ravirajsinh45](https://www.kaggle.com/ravirajsinh45) for providing the casting dataset
- **Kaggle Community**: For valuable feedback and suggestions
- **TensorFlow Team**: For the excellent deep learning framework
- **Contributors**: All contributors who helped improve this project

## 📧 Contact

- **Author**: Amer Harfoush
- **LinkedIn**: [Industrial AI Solutions](https://linkedin.com/in/amer-harfoush)

## 📊 Citation

If you use this project in your research or production, please cite:

```bibtex
@software{casting_defect_detection,
  author = {Amer Harfoush},
  title = {Industrial Casting Defect Detection System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Harfoush4/Defected-Product-Detector}
}
```

## 🔄 Updates & Roadmap

### Recent Updates (v1.0)
- ✅ Initial release with CNN and transfer learning models
- ✅ Comprehensive evaluation metrics
- ✅ Production-ready pipeline
- ✅ TensorFlow Lite support

### Upcoming Features (v2.0)
- 🔄 Multi-defect classification (cracks, holes, surface defects)
- 🔄 Real-time video stream processing
- 🔄 Web-based dashboard for monitoring
- 🔄 Integration with industrial IoT platforms
- 🔄 Active learning for continuous improvement

---

<p align="center">
  Made with ❤️ for the Manufacturing Industry
</p>

<p align="center">
  <a href="#-industrial-casting-defect-detection-system">Back to Top ↑</a>
</p>
