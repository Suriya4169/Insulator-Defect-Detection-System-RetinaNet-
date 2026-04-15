<div align="center">

# 🔌 FedRetinaNet — Privacy-Preserving Insulator Defect Detection via Federated Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: IDD-CPLID](https://img.shields.io/badge/Dataset-IDD--CPLID-blue)](https://universe.roboflow.com/project-vjqdt/idd-cplid)

**A Federated Learning framework for privacy-preserving insulator defect detection on high-voltage power-line infrastructure using RetinaNet with multi-backbone support.**

[Overview](#-overview) •
[Architecture](#-architecture) •
[Results](#-results) •
[Setup](#-setup) •
[Usage](#-usage) •
[Team](#-team)

</div>

---

## 📋 Overview

Automated inspection of insulator defects on power-line infrastructure is critical for preventing grid failures. Traditional centralized deep learning approaches require pooling sensitive inspection data from multiple utility providers into a single server — raising **privacy**, **regulatory**, and **bandwidth** concerns.

This project proposes a **Federated Learning (FL)** approach where multiple simulated clients collaboratively train a shared **RetinaNet** object detection model **without exchanging raw image data**. Each client trains locally on its data partition, and only model weight updates are aggregated on a central server using **Weighted Federated Averaging (FedAvg)**.

### Key Contributions

- **Privacy-Preserving Training**: Simulated multi-client federated setup (3 clients) where raw data never leaves the client.
- **Dual-Backbone Comparison**: Experiments with both **ResNet-50 FPN V2** (high-accuracy) and **MobileNetV3** (lightweight/edge) backbones.
- **Advanced Data Augmentation Pipeline**: Heavy augmentation with **Copy-Paste defect synthesis**, geometric transforms, color jittering, cutout, and a **defect crop bank** for class-imbalance mitigation.
- **Custom Focal Loss Tuning**: Per-class alpha weighting (`[0.25, 0.75, 0.25]`) to prioritize the under-represented defect class.
- **Comprehensive Evaluation**: COCO-style AP@50/AP@75, confusion matrices, precision-recall curves, and per-class accuracy tracking.

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    CENTRAL SERVER                        │
│                                                         │
│   ┌───────────────────────────────────────────────┐     │
│   │         Global RetinaNet Model                │     │
│   │    (ResNet-50 FPN V2 / MobileNetV3)           │     │
│   └──────────────────┬────────────────────────────┘     │
│                      │                                  │
│           Weighted FedAvg Aggregation                   │
│          ┌───────────┼───────────┐                      │
│          │           │           │                      │
└──────────┼───────────┼───────────┼──────────────────────┘
           │           │           │
    ┌──────▼──┐  ┌─────▼───┐  ┌───▼──────┐
    │Client 1 │  │Client 2 │  │Client 3  │
    │Local    │  │Local    │  │Local     │
    │Training │  │Training │  │Training  │
    │         │  │         │  │          │
    │Partition│  │Partition│  │Partition │
    │  1/3    │  │  2/3    │  │  3/3     │
    └─────────┘  └─────────┘  └──────────┘
```

### Model Architecture — RetinaNet

| Component | Specification |
|---|---|
| **Detector** | RetinaNet (anchor-based, single-stage) |
| **Backbone (Primary)** | ResNet-50 + FPN V2 (pre-trained on COCO) |
| **Backbone (Lightweight)** | MobileNetV3-Large + FPN |
| **Detection Head** | `RetinaNetHead` with `BatchNorm2d` normalization |
| **Loss Function** | Focal Loss (α = `[0.25, 0.75, 0.25]`, γ = 2) |
| **Anchor Generator** | Default multi-scale FPN anchors |
| **Classes** | 3 — `background (0)`, `defect (1)`, `insulator (2)` |

### Federated Learning Configuration

| Parameter | Value |
|---|---|
| Number of Clients | 3 |
| Communication Rounds | 8 |
| Client Local Epochs | 2 per round |
| Aggregation Strategy | Weighted FedAvg (proportional to dataset size) |
| Optimizer (Client) | AdamW (`lr=3e-4`, `weight_decay=1e-4`) |
| LR Scheduler | OneCycleLR (cosine annealing, `pct_start=0.15`) |
| Mixed Precision | FP16 via `torch.cuda.amp` |
| Early Stopping | Patience = 3, δ = 0.005 on mAP@50 |

---

## 📊 Dataset — IDD-CPLID

The **Insulator Defect Detection - Chinese Power Line Insulator Dataset (IDD-CPLID)** is sourced from [Roboflow Universe](https://universe.roboflow.com/project-vjqdt/idd-cplid) under the **CC BY 4.0** license.

| Split | Images | Annotation Format |
|---|---|---|
| **Train** | ~2,200 (×7 augmented = ~15,400 source variants) | COCO JSON |
| **Validation** | ~250 | COCO JSON |
| **Test** | ~200 | COCO JSON |
| **Total** | **3,203 images** | — |

### Pre-Processing (Applied via Roboflow)
- Auto-orientation with EXIF stripping
- Resized to **640×640** (stretch)

### Augmentation (Applied During Training — 7 versions per source image)
- 50% horizontal flip
- Random crop (0–20%)
- Random rotation (±15°)
- Random shear (±10° H/V)

### Additional Runtime Augmentations (Custom Pipeline)
- Vertical flip (30%), 90° rotation (30%)
- Scale + crop (60%), color jitter (70%)
- Gaussian blur (25%), grayscale (10%)
- Brightness/contrast adjustment (50%)
- Cutout / random erasing (40%)
- Sharpness enhancement (30%)
- **Copy-Paste Defect Synthesis** (35%) — pastes cropped defect patches from a pre-built bank of 300 crops onto training images

---

## 📈 Results

### Centralized Baseline (ResNet-50 FPN V2)

| Metric | Defect | Insulator | Mean |
|---|---|---|---|
| **AP@50** | **1.00** | **0.88** | **0.94** |

### Federated Learning (ResNet-50 FPN V2, 3 Clients, 8 Rounds)

| Metric | Defect | Insulator | Mean |
|---|---|---|---|
| **AP@50** | **0.99** | **0.83** | **0.91** |

### Key Observations

- The federated model retains **~97%** of the centralized baseline's mAP@50 performance while keeping data decentralized.
- The **defect class** achieves near-perfect detection (AP@50 = 0.99) thanks to focal loss tuning and copy-paste augmentation.
- The **insulator class** shows a modest 5-point AP drop under federation, likely due to data heterogeneity across client partitions.

### Precision-Recall Curves

| Baseline (Centralized) | Federated Learning |
|---|---|
| Defect AP = 1.00, Insulator AP = 0.88 | Defect AP = 0.99, Insulator AP = 0.83 |

> Precision-Recall curve plots are available in `Retinanet_resnet_backbone/Base_model_with_resnet_backbone/precision_recall_baseline.png` and `Retinanet_resnet_backbone/Fed_learning_model_with_resnet_backbone/P-vs-R-fedlearning.png`.

---

## 🚀 Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.7+ (recommended)
- 8 GB+ GPU VRAM (16 GB recommended for batch_size=16)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/FedRetinaNet-Insulator-Defect-Detection.git
cd FedRetinaNet-Insulator-Defect-Detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy tqdm Pillow
```

### Dataset Setup

1. Download the IDD-CPLID dataset from [Roboflow](https://universe.roboflow.com/project-vjqdt/idd-cplid) in **COCO format**.
2. Place the dataset in the project root:

```
Dataset - IDD-CPLID.v3-cplid_new.coco/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

3. Update the `DATA_PATH` in `FederatedConfig` (or `Config` for baseline) to point to your dataset directory.

---

## 💻 Usage

### Train Baseline (Centralized)

```bash
cd Insulator-Defect-Detection-System-RetinaNet-/Retinanet_resnet_backbone/Base_model_with_resnet_backbone/retinanet_code
python train.py
```

### Train Federated Model

```bash
cd Insulator-Defect-Detection-System-RetinaNet-/Retinanet_resnet_backbone/Fed_learning_model_with_resnet_backbone/Fed_learning_Code
python federated_train.py
```

### Train MobileNet Backbone (Hybrid — from Notebooks)

```bash
cd Insulator-Defect-Detection-System-RetinaNet-/Retinanet_with_mobilenet_backbone(hybrid)
# Open and run the Jupyter notebooks:
#   - Mobilenet(backbone)-retinanet.ipynb   → Centralized baseline
#   - Fed_learning-MB-Retinanet.ipynb       → Federated variant
```

### Generate Federated Evaluation Report

```bash
cd Insulator-Defect-Detection-System-RetinaNet-/Retinanet_resnet_backbone/Fed_learning_model_with_resnet_backbone/Fed_learning_Code
python generate_federated_report.py
```

---

## 📁 Project Structure

```
FedRetinaNet-Insulator-Defect-Detection/
│
├── Dataset - IDD-CPLID.v3-cplid_new.coco/   # Dataset (COCO format)
│   ├── train/
│   ├── valid/
│   └── test/
│
├── Insulator-Defect-Detection-System-RetinaNet-/
│   │
│   ├── Retinanet_resnet_backbone/
│   │   ├── Base_model_with_resnet_backbone/
│   │   │   ├── retinanet_code/
│   │   │   │   ├── model.py              # RetinaNet + ResNet50 FPN V2
│   │   │   │   ├── dataset.py            # COCO dataset loader
│   │   │   │   ├── engine.py             # Train/eval engine
│   │   │   │   ├── train.py              # Centralized training script
│   │   │   │   ├── evaluate.py           # Standalone evaluation
│   │   │   │   ├── transforms.py         # Data transforms
│   │   │   │   ├── util.py               # GPU utilities
│   │   │   │   └── notebooks/
│   │   │   │       └── Walkthrough.ipynb  # Step-by-step notebook
│   │   │   └── precision_recall_baseline.png
│   │   │
│   │   └── Fed_learning_model_with_resnet_backbone/
│   │       ├── Fed_learning_Code/
│   │       │   ├── model.py              # RetinaNet + custom focal loss alpha
│   │       │   ├── dataset.py            # FL dataset with Copy-Paste augmentation
│   │       │   ├── engine.py             # FL-aware train/eval engine
│   │       │   ├── federated_train.py    # Main FL training loop (FedAvg)
│   │       │   ├── generate_federated_report.py  # Report generator
│   │       │   ├── transforms.py         # Data transforms
│   │       │   └── check_cuda.py         # CUDA availability check
│   │       └── P-vs-R-fedlearning.png
│   │
│   └── Retinanet_with_mobilenet_backbone(hybrid)/
│       ├── Mobilenet(backbone)-retinanet.ipynb     # Centralized + MobileNet
│       └── Fed_learning-MB-Retinanet.ipynb         # Federated variant
│
├── checkpoints/                           # Saved model weights
├── best_federated_report_r6.pdf           # Auto-generated FL report
├── professional_evaluation_report.pdf     # Professional evaluation report
└── README.md                              # This file
```

---

## ⚙️ Hyperparameter Reference

<details>
<summary>Click to expand full hyperparameter table</summary>

| Category | Parameter | Baseline | Federated |
|---|---|---|---|
| **Optimizer** | Type | SGD | AdamW |
| | Learning Rate | 5e-4 | 3e-4 |
| | Weight Decay | 5e-4 | 1e-4 |
| | Momentum | 0.93 | — |
| **Scheduler** | Type | CosineAnnealingLR | OneCycleLR |
| | T_max / pct_start | 100 epochs | 0.15 |
| **Training** | Epochs | 100 | 8 rounds × 2 epochs |
| | Batch Size | 8 | 16 |
| | Gradient Clipping | 1.0 | 1.0 |
| | Mixed Precision | ✅ | ✅ |
| | Early Stopping | patience=20 | patience=3 |
| **Augmentation** | Copy-Paste | ❌ | ✅ (prob=0.35) |
| | Defect Bank Size | — | 300 crops |
| | Weighted Sampler | ❌ | ✅ (weight=4.0) |
| **Focal Loss** | Alpha | default | [0.25, 0.75, 0.25] |

</details>

---

## 🔬 Technical Highlights

1. **Weighted FedAvg Aggregation**: Client model weights are averaged proportionally to each client's dataset size, ensuring clients with more data have proportionally more influence on the global model.

2. **Copy-Paste Defect Augmentation**: A bank of 300 defect crops is pre-extracted from training data. During training, 1–3 randomly scaled defect patches are pasted onto images with 35% probability, drastically increasing effective defect samples.

3. **Class-Balanced Focal Loss**: The focal loss alpha vector `[0.25, 0.75, 0.25]` assigns 3× more weight to the minority defect class, addressing the inherent class imbalance in infrastructure inspection datasets.

4. **OneCycleLR Scheduling**: Each client uses a per-round OneCycleLR schedule with cosine annealing, avoiding learning rate staleness across federated rounds.

---

## 👥 Team

| Name | Role |
|---|---|
| **Jeevakamal K R** | Team Member |
| **Jeiesh J S** | Team Member |
| **Suriya Dharsuan K G** | Team Member |
| **Prathap P** | Team Member |

---

## 📚 References

1. Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." *ICCV*, 2017.
2. McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*, 2017.
3. Tao, X., et al. "Detection of Power Line Insulator Defects Using Aerial Images Analyzed with Convolutional Neural Networks." *IEEE T-SMC*, 2020.
4. IDD-CPLID Dataset — Roboflow Universe. https://universe.roboflow.com/project-vjqdt/idd-cplid

---

## 📄 License

This project is licensed under the **MIT License**. The IDD-CPLID dataset is provided under the **CC BY 4.0** license.

---

<div align="center">
<sub>Built with ❤️ as a semester course project</sub>
</div>