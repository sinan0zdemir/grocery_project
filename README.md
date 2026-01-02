# Grocery Detection & Classification Project

A comprehensive computer vision system for detecting and classifying grocery products using YOLO object detection and ArcFace-based product classification.

## Overview

This project combines two main components:
- **Detection**: YOLO-based model for detecting grocery products in images
- **Classification**: ArcFace with ResNet-34 backbone for fine-grained product recognition via embedding similarity

The system is designed to work with the Grocery_products dataset, supporting annotation-based evaluation and cropped query visualization.

## Project Structure

```
grocery_project/
├── detection/
│   ├── inference/
│   │   ├── detect_shelf.py          # Detection inference
│   │   ├── crop_products.py         # Crop detected products
│   │   ├── detection_inference.py   # Alternative inference script
│   │   ├── detection_output/        # CSV results
│   │   └── inference_dataset/       # Test images
│   ├── eval/
│   │   ├── eval_yolo11_SKU110K.py   # SKU110K evaluation
│   │   └── eval_yolo11_grocery.py   # Grocery evaluation
│   ├── training/
│   │   └── detection_train_yolov11.ipynb
│   └── weights/
│       └── weights_11S_new.pt       # YOLOv11 weights
├── classification/
│   ├── training/
│   │   ├── arcface_grocery_fixed.ipynb
│   │   └── arcface_grocery_fixed.json
│   ├── eval/
│   │   ├── evaluate_arcface.py      # Evaluation script
│   │   └── outputs/                 # Metrics & visualizations
│   └── checkpoints/
│       └── best.pth                 # Trained ArcFace model
├── datasets/
│   ├── Grocery_products/
│   │   ├── Training/                # Reference images
│   │   ├── Testing/                 # Store images
│   │   └── Annotations/             # CSV annotations
│   └── SKU110K/                     # SKU detection dataset
├── main.py
├── pyproject.toml
└── README.md
```

## Features

- **Product Detection**: Localize grocery products in shelf images using YOLOv11
- **ArcFace Classification**: Embedding-based product recognition with cosine similarity
- **Annotation-based Evaluation**: Uses CSV annotations for ground truth matching
- **Cropped Query Visualization**: Output images show cropped query products vs top-5 matches
- **Reference Database Caching**: Pre-computed embeddings saved to `reference_db.pt` for faster re-runs

## Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended)

### Using `uv` (Fast & Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

### 1. Object Detection

```bash
python detection/inference/detect_shelf.py
```

### 2. Product Classification (Evaluation)

```bash
python classification/eval/evaluate_arcface.py
```

**What it does:**
1. Parses annotations from `datasets/Grocery_products/Annotations/`
2. Builds a filtered reference database from Training images (cached to `reference_db.pt`)
3. Evaluates on test samples using bounding box crops
4. Outputs Top-1/Top-5 accuracy to `classification/eval/outputs/metrics_fixed.txt`
5. Generates visualization images with cropped queries vs top-5 predictions

**Expected Output:**
```
Top-1 Accuracy: ~81%
Top-5 Accuracy: ~90%
```

## Model Details

### Detection Model
- **Framework**: YOLOv11
- **Output**: Bounding boxes with confidence scores

### Classification Model
- **Architecture**: ResNet-34 backbone + ArcFace head
- **Embedding Dim**: 512
- **Training**: ArcFace loss with Hierarchical Auxiliary Loss (HAL)
- **Inference**: Cosine similarity against reference embeddings

## Configuration

Key settings in `classification/eval/evaluate_arcface.py`:
- `BATCH_SIZE`: 32
- `TOP_K`: 5 (for top-k accuracy)
- `NUM_EXAMPLE_IMAGES`: 5 (visualizations to generate)

## Output Files

| File | Description |
|------|-------------|
| `classification/eval/outputs/metrics_fixed.txt` | Top-1 and Top-5 accuracy |
| `classification/eval/outputs/reference_db.pt` | Cached reference embeddings |
| `classification/eval/outputs/viz_*.png` | Cropped query vs top-5 predictions |

