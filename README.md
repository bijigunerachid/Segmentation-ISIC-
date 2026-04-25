# ISIC Skin Lesion Segmentation

Pixel-wise segmentation of dermoscopic images using deep learning, trained and evaluated on the **ISIC 2018 Task 1** benchmark dataset.

Three architectures are implemented and compared: **U-Net**, **SegNet**, and **DeepLabV3** (ResNet-50 backbone).

---

## Results on Test Set (390 images)

| Model | Dice | IoU | Precision | Recall | Params | Epochs |
|-------|------|-----|-----------|--------|--------|--------|
| U-Net | 0.8664 | 0.7899 | 0.9014 | 0.8802 | 31.0 M | 20 |
| SegNet | 0.8575 | 0.7762 | 0.8815 | 0.8832 | 24.9 M | 30 |
| **DeepLabV3** | **0.9030** | **0.8263** | **0.9251** | **0.9186** | 39.6 M | 47 |

DeepLabV3 achieves the best overall performance with a Dice score of **0.9030**.

---

## Dataset

**ISIC 2018 Task 1 — Lesion Boundary Segmentation**

- 2,594 dermoscopic images with binary segmentation masks
- Split: 1,815 train / 389 val / 390 test
- Input size: 256 x 256 pixels
- Task: binary pixel classification (lesion vs. background)

The dataset is not included in this repository (too large). Download it from the [ISIC Archive](https://challenge.isic-archive.com/data/#2018) and place it in:

```
data/
  images/         # dermoscopic images (.jpg)
  masques/        # binary masks (.png)
```

---

## Project Structure

```
ISIC_Project/
├── streamlit_app.py          # Web application (4 pages)
├── test_models.py            # Standalone CLI script — outputs prediction grid
├── app_segmentation.py       # Tkinter desktop app
│
├── src/
│   ├── dataset.py            # PyTorch Dataset + augmentations
│   ├── model.py              # U-Net architecture
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation utilities
│   └── utils.py              # Dice, IoU metric functions
│
├── notebooks/
│   ├── unet/                 # U-Net training notebook
│   ├── segnet/               # SegNet training notebook
│   └── deeplabv3/            # DeepLabV3 training notebook
│
├── outputs/
│   ├── test_metrics.json             # U-Net test results
│   ├── segnet/conclusion_segnet.json
│   ├── deeplabv3/test_metrics.json
│   ├── training_curves.png
│   └── confusion_matrix/
│
├── models/
│   └── best_unet_isic.pth    # U-Net checkpoint (not in git — see below)
│
└── requirements.txt
```

> **Model checkpoints** (`.pth` files) are excluded from git due to size (96 MB to 464 MB).
> See the **Model Weights** section below.

---

## Installation

```bash
git clone https://github.com/bijigunerachid/Segmentation-ISIC-.git
cd Segmentation-ISIC-
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.0+ recommended.

---

## Model Weights

The trained checkpoints must be placed at these paths:

| Model | Path |
|-------|------|
| U-Net | `models/best_unet_isic.pth` |
| SegNet | `outputs/segnet/best_segnet.pth` |
| DeepLabV3 | `outputs/deeplabv3/best_deeplabv3_isic.pth` |

**Option A — Train yourself** using the Jupyter notebooks in `notebooks/`.

**Option B — Download pre-trained weights** from Hugging Face Hub:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="bijigunerachid/isic-segmentation-models",
                filename="best_unet_isic.pth", local_dir="models/")

hf_hub_download(repo_id="bijigunerachid/isic-segmentation-models",
                filename="best_segnet.pth", local_dir="outputs/segnet/")

hf_hub_download(repo_id="bijigunerachid/isic-segmentation-models",
                filename="best_deeplabv3_isic.pth", local_dir="outputs/deeplabv3/")
```

---

## Streamlit Web Application

The app has four pages:

| Page | Description |
|------|-------------|
| **Home** | Project overview, model cards, dataset statistics |
| **Prediction** | Run inference on one image with a selected model |
| **Comparison** | Run all 3 models side-by-side on the same image |
| **Metrics** | Pre-computed test results, training curves, confusion matrices |

Both **Prediction** and **Comparison** include a **Demo Gallery** — 8 pre-generated synthetic dermoscopic images. Click any thumbnail to run inference instantly (no real dataset required).

### Run locally

```bash
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Add your Hugging Face repo in the **Secrets** section of the app settings:

```toml
HF_REPO = "bijigunerachid/isic-segmentation-models"
HF_TOKEN = "hf_..."   # only needed if the repo is private
```

The app will auto-download all three model checkpoints on first launch.

---

## CLI Test Script

Generate a side-by-side prediction grid for all three models without launching the web app:

```bash
# 4 synthetic demo images (no dataset needed)
python test_models.py

# Use real images from a folder
python test_models.py --images data/images --n 6

# With overlay columns and high DPI
python test_models.py --overlay --dpi 200
```

Output is saved to `outputs/predictions/test_predictions_grid.png`.

```
Options:
  --images DIR     Folder of dermoscopic images (default: synthetic demo)
  --n INT          Number of images to process (default: 4)
  --seed INT       Seed for synthetic demo images (default: 0)
  --threshold F    Segmentation threshold 0-1 (default: 0.5)
  --overlay        Add overlay columns to the grid
  --dpi INT        Output resolution in DPI (default: 150)
  --output PATH    Override output file path
```

---

## Training

Each model has a dedicated Jupyter notebook:

```bash
jupyter notebook notebooks/unet/01_Train_UNet.ipynb
jupyter notebook notebooks/segnet/01_Train_SegNet.ipynb
jupyter notebook notebooks/deeplabv3/01_Train_DeepLabV3.ipynb
```

**Training configuration:**

| Setting | Value |
|---------|-------|
| Image size | 256 x 256 |
| Batch size | 8 |
| Loss | Dice + Binary Cross-Entropy |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Augmentations | Flip, rotate, brightness, elastic deformation (Albumentations) |

GPU training recommended (Google Colab or local CUDA). See `SETUP_COLAB.md` for the full Google Colab + Drive setup guide.

---

## Model Architectures

### U-Net
Encoder-decoder with skip connections. The encoder progressively halves spatial resolution while doubling feature channels (64 to 1024). Skip connections concatenate encoder feature maps into the decoder, preserving fine-grained spatial detail.

### SegNet
Encoder-decoder that reuses max-pooling indices from the encoder for precise upsampling in the decoder — no skip connections. Lighter memory footprint than U-Net.

### DeepLabV3 (ResNet-50 backbone)
Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale context using dilated convolutions at multiple rates. Pre-trained ImageNet ResNet-50 backbone fine-tuned end-to-end on ISIC data. Best-performing model in this study.

---

## Requirements

```
torch
torchvision
albumentations
opencv-python-headless
numpy  matplotlib  scikit-learn  scipy  pandas  tqdm  Pillow
streamlit
huggingface_hub
```

---

## License

This project is for academic and research purposes only.
Dataset: [ISIC 2018 Challenge](https://challenge.isic-archive.com/data/#2018) — subject to ISIC terms of use.
This tool is not intended for clinical diagnosis.
