# Grocery Detection & Classification Project

A comprehensive computer vision system for detecting and classifying grocery products using YOLO object detection and deep learning-based product classification.

## Overview

This project combines two main components:
- **Detection**: YOLO-based model for detecting grocery products in images
- **Classification**: Deep learning models (VGG16 with MAC pooling) for fine-grained product classification

The system is designed to work with both studio and in-store grocery product images, supporting hierarchy-based classification across multiple product categories.

## Project Structure

```
grocery_project/
├── detection/          # YOLO object detection module
│   ├── best.pt        # Pre-trained YOLO model
│   ├── inference_yolo.py
│   ├── crop_products.py
│   └── detection_output/  # Detection results (CSVs)
├── classification/     # Product classification module
│   ├── dihe_pytorch.py    # Classification model (DIHE with VGG16)
│   ├── dihe_train.py      # Training script
│   ├── test_dihe.py       # Testing script
│   └── data/              # Training data
│       ├── studio/        # Studio product images
│       └── instore/       # In-store product images
├── main.py            # Main entry point
├── pyproject.toml     # Project dependencies
└── README.md          # This file
```

## Features

- **Product Detection**: Localize grocery products in images using YOLO
- **Product Classification**: Classify detected products across 60+ brands and categories
- **Multi-Domain Learning**: Handle both studio and in-store images
- **Hierarchy-based Classification**: Support for macro-categories and fine-grained product classes
- **Batch Processing**: Process multiple images with CSV output results

## Installation

### Prerequisites
- Python 3.12 or higher
- pip or `uv` (recommended)

### Using `uv` (Fast & Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv sync

# Or install specific packages
uv pip install torch torchvision ultralytics opencv-python
```

### Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or directly from pyproject.toml
pip install .
```

## Usage

### 1. Object Detection

Detect grocery products in images:

```bash
cd detection/
python inference_yolo.py
```

**Configuration** (in `inference_yolo.py`):
- `MODEL_PATH`: Path to YOLO model (default: `./best.pt`)
- `INPUT_FOLDER`: Directory containing test images
- `OUTPUT_FOLDER`: Directory for detection results
- `SCORE_THRESHOLD`: Confidence threshold (0.5 recommended)

**Output**: CSV files with detected bounding boxes and confidence scores

### 2. Product Classification

Train or test the classification model:

```bash
# Train the model
cd classification/
python dihe_train.py

# Test the model
python test_dihe.py
```


## Dependencies

Key dependencies (see `pyproject.toml` for complete list):
- `torch` & `torchvision`: Deep learning framework
- `ultralytics`: YOLO object detection
- `opencv-python`: Image processing
- `pandas`: Data handling
- `numpy`: Numerical computing
- `pillow`: Image operations

## Dataset

The project expects the following data structure:

```
data/
├── studio/        # Controlled environment product images
│   └── Brand/     # Brand-specific folders
│       └── product_images.jpg
└── instore/       # Real store product images
    └── product_images.jpg
```

## Model Details

### Detection Model
- **Framework**: YOLOv8
- **Input**: Product images
- **Output**: Bounding boxes with confidence scores

### Classification Model
- **Architecture**: VGG16 with MAC (Maximum Activation of Convolutions) pooling
- **Features**: 
  - L2 normalization for feature stability
  - Hierarchy-aware learning
  - Domain adaptation for studio vs. in-store images
- **Classes**: 60+ grocery product brands and categories

## Configuration

Adjust settings in individual scripts:
- Detection threshold in `detection/inference_yolo.py`
- Model paths and hyperparameters in `classification/dihe_train.py`
- Data paths in `main.py`

## Output

Results are saved as CSV files in `detection/detection_output/`:
- Bounding box coordinates (x_min, y_min, x_max, y_max)
- Confidence scores
- Class predictions
