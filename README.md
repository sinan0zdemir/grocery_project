# Grocery Detection & Classification Project

A comprehensive computer vision system for detecting and classifying grocery products using YOLO object detection and embedding-based product classification (ArcFace, DIHE).

## Overview

This project combines two main components:
- **Detection**: YOLOv11-based model for detecting grocery products in shelf images
- **Classification**: ArcFace and DIHE models for fine-grained product recognition via embedding similarity

## Project Structure

```
grocery_project/
├── detection/
│   ├── inference/                      # Detection inference scripts
│   ├── eval/
│   │   ├── eval_yolo11_grocery.py      # Grocery Products evaluation
│   │   ├── eval_yolo11_SKU110K.py      # SKU110K evaluation
│   │   ├── eval_yolo11_SDP.py          # SDP_Product evaluation
│   │   ├── grocery_results/            # Grocery eval outputs
│   │   └── sdp_results/                # SDP eval outputs
│   ├── training/
│   │   └── detection_train_yolov11.ipynb
│   └── weights/
│       ├── weights_11S_new.pt          # YOLOv11-S weights (latest)
│       └── weights_11s.pt              # YOLOv11-S weights
├── classification/
│   ├── training/
│   │   ├── arcface_grocery_fixed.ipynb # ArcFace training notebook
│   │   ├── dihe.ipynb                  # DIHE training notebook
│   │   └── dihe.json                   # DIHE notebook (Colab compatible)
│   ├── eval/
│   │   ├── evaluate_arcface.py         # ArcFace evaluation
│   │   ├── evaluate_dihe.py            # DIHE evaluation
│   │   ├── visualize_embeddings.py     # UMAP embedding visualization
│   │   └── outputs/                    # Metrics & visualizations
│   └── checkpoints/
│       ├── best_2.pth                  # ArcFace model (286 classes)
│       ├── best.pth                    # ArcFace model (older)
│       ├── best_all_cat.pth            # ArcFace model (all categories)
│       └── epoch_9.tar                 # DIHE encoder weights
├── datasets/
│   ├── Grocery_products/
│   │   ├── Training/                   # Reference product images
│   │   ├── Testing/                    # Store shelf images
│   │   └── Annotations/                # CSV annotations (s1_10.csv format)
│   ├── SKU110K/                        # SKU detection dataset
│   └── SDP_Product.v1/                 # Roboflow product dataset
├── main.py
├── pyproject.toml
└── README.md
```

## Features

- **Multi-Dataset Detection Evaluation**: Support for Grocery_products, SKU110K, and SDP_Product datasets
- **Multiple Classification Models**: ArcFace and DIHE embedders
- **Embedding Visualization**: UMAP projection with category-based coloring
- **Reference Database Caching**: Pre-computed embeddings saved for faster re-evaluation
- **Cropped Query Visualization**: Output images show cropped query products vs top-k matches

## Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended for training, CPU works for inference)

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

### Detection Evaluation

**Grocery Products Dataset:**
```bash
python detection/eval/eval_yolo11_grocery.py
```

**SKU110K Dataset:**
```bash
python detection/eval/eval_yolo11_SKU110K.py
```

**SDP_Product Dataset (Roboflow format):**
```bash
python detection/eval/eval_yolo11_SDP.py
```

### Classification Evaluation

**ArcFace Model:**
```bash
python classification/eval/evaluate_arcface.py
```

**DIHE Model:**
```bash
python classification/eval/evaluate_dihe.py --encoder-weights classification/checkpoints/epoch_9.tar
```

### Visualize Embeddings

```bash
python classification/eval/visualize_embeddings.py
```

## Model Checkpoints

### Detection (YOLOv11)
| File | Description |
|------|-------------|
| `detection/weights/weights_11S_new.pt` | YOLOv11-S trained on Grocery Products |

### Classification
| File | Description |
|------|-------------|
| `classification/checkpoints/best_2.pth` | ArcFace ResNet-34 (286 classes, 83.3% Top-1) |
| `classification/checkpoints/epoch_9.tar` | DIHE VGG16 encoder |

## Evaluation Results

### ArcFace Classification (Grocery Products)
```
Top-1 Accuracy: 83.31%
Top-5 Accuracy: 92.28%
```

### Detection (YOLOv11)
See `detection/eval/*/metrics_summary.png` for mAP curves.

## Output Files

| File | Description |
|------|-------------|
| `classification/eval/outputs/metrics_fixed.txt` | ArcFace evaluation metrics |
| `classification/eval/outputs/dihe_results.txt` | DIHE evaluation metrics |
| `classification/eval/outputs/reference_db.pt` | Cached ArcFace embeddings |
| `classification/eval/outputs/embeddings_visualization.png` | UMAP projection |
| `classification/eval/outputs/viz_*.png` | Query vs top-5 predictions |
| `detection/eval/grocery_results/` | Detection evaluation outputs |
| `detection/eval/sdp_results/` | SDP detection evaluation outputs |

## Command-Line Options

### evaluate_arcface.py
No arguments required - uses default paths.

### evaluate_dihe.py
```bash
--encoder-weights PATH   # Path to .tar encoder weights
--model-type TYPE        # vgg16 or resnet50
--batch-size N           # Batch size (default: 8)
--k VALUES               # Top-k values, comma-separated (default: 1,5,10)
```

### eval_yolo11_SDP.py
```bash
--weights PATH           # YOLO weights path
--dataset PATH           # Dataset root path
--conf FLOAT             # Confidence threshold (default: 0.25)
--num-examples N         # Example images to generate
```

## Dependencies

Tüm bağımlılıkları kurmak için:

```bash
pip3 install -r requirements.txt
```

| Paket | Açıklama |
|-------|----------|
| `torch` | Derin öğrenme framework'ü |
| `torchvision` | PyTorch görüntü modelleri ve dönüşümleri |
| `ultralytics` | YOLOv11 nesne tespiti |
| `opencv-python` | Görüntü işleme (cv2) |
| `Pillow` | Görüntü okuma/yazma (PIL) |
| `numpy` | Sayısal hesaplama |
| `pandas` | Veri işleme |
| `matplotlib` | Görselleştirme |
| `scipy` | Bilimsel hesaplama (sinyal işleme, istatistik) |
| `tqdm` | İlerleme çubuğu |
| `fastapi` | Web API framework'ü |
| `uvicorn[standard]` | ASGI sunucusu |
| `python-multipart` | Dosya yükleme desteği |
| `jinja2` | HTML şablon motoru |
| `cvpce` | DIHE kütüphanesi (`../cvpce/` dizininden) |
| `umap-learn` | Embedding görselleştirme (UMAP projeksiyonu) |
