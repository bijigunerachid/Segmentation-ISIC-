"""
test_models.py
--------------
Run U-Net, SegNet and DeepLabV3 on a set of images and save a comparison
grid similar to outputs/predictions/test_predictions.png.

Usage
-----
# Use synthetic demo images (no dataset needed):
    python test_models.py

# Use real images from a folder:
    python test_models.py --images data/images --n 6

# Show overlay columns as well:
    python test_models.py --overlay

# Full options:
    python test_models.py --help
"""

import argparse
import os
import sys
import glob
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Defaults ──────────────────────────────────────────────────────────────────
IMG_SIZE      = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
THRESHOLD     = 0.5
DPI           = 150
OUTPUT_PATH   = os.path.join(ROOT, "outputs", "predictions", "test_predictions_grid.png")

CHECKPOINTS = {
    "U-Net":     os.path.join(ROOT, "models",  "best_unet_isic.pth"),
    "SegNet":    os.path.join(ROOT, "outputs", "segnet",    "best_segnet.pth"),
    "DeepLabV3": os.path.join(ROOT, "outputs", "deeplabv3", "best_deeplabv3_isic.pth"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
# Model architectures  (must match saved checkpoint keys exactly)
# ════════════════════════════════════════════════════════════════════════════

class _DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
        )
    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList(); self.ups = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        ic = n_channels
        for f in features: self.downs.append(_DoubleConv(ic, f)); ic = f
        self.bottleneck = _DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(_DoubleConv(f*2, f))
        self.final = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for d in self.downs: x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i//2]
            if x.shape != s.shape: x = F.interpolate(x, size=s.shape[2:])
            x = self.ups[i+1](torch.cat([s, x], 1))
        return self.final(x)


class _SBlock(nn.Module):
    def __init__(self, ic, oc, n=2):
        super().__init__()
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(ic if i==0 else oc, oc, 3, padding=1),
                       nn.BatchNorm2d(oc), nn.ReLU(True)]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1=_SBlock(in_channels,64,2); self.enc2=_SBlock(64,128,2)
        self.enc3=_SBlock(128,256,3);        self.enc4=_SBlock(256,512,3)
        self.enc5=_SBlock(512,512,3)
        self.dec5=_SBlock(512,512,3); self.dec4=_SBlock(512,256,3)
        self.dec3=_SBlock(256,128,3); self.dec2=_SBlock(128,64,2)
        self.dec1=_SBlock(64,64,2)
        self.pool=nn.MaxPool2d(2,2,return_indices=True)
        self.unpool=nn.MaxUnpool2d(2,2)
        self.final=nn.Conv2d(64,out_channels,1)

    def forward(self, x):
        x1=self.enc1(x); x,i1=self.pool(x1)
        x2=self.enc2(x); x,i2=self.pool(x2)
        x3=self.enc3(x); x,i3=self.pool(x3)
        x4=self.enc4(x); x,i4=self.pool(x4)
        x5=self.enc5(x); x,i5=self.pool(x5)
        x=self.unpool(x,i5,x5.size()); x=self.dec5(x)
        x=self.unpool(x,i4,x4.size()); x=self.dec4(x)
        x=self.unpool(x,i3,x3.size()); x=self.dec3(x)
        x=self.unpool(x,i2,x2.size()); x=self.dec2(x)
        x=self.unpool(x,i1,x1.size()); x=self.dec1(x)
        return self.final(x)


class DeepLabV3(nn.Module):
    def __init__(self, out_channels=1, pretrained=False):
        super().__init__()
        weights = seg_models.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = seg_models.deeplabv3_resnet50(weights=weights, aux_loss=True)
        self.model.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.model(x)["out"]
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_model(name: str, ckpt_path: str) -> nn.Module:
    print(f"  Loading {name} from {os.path.relpath(ckpt_path, ROOT)} ...", end=" ", flush=True)
    if name == "U-Net":
        model = UNet()
    elif name == "SegNet":
        model = SegNet()
    elif name == "DeepLabV3":
        model = DeepLabV3(pretrained=False)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.load_state_dict(load_checkpoint(ckpt_path), strict=True)
    model.to(DEVICE).eval()
    print("OK")
    return model


def preprocess(pil_img: Image.Image):
    arr  = np.array(pil_img.convert("RGB"))
    orig = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t    = _tf(image=arr)["image"].unsqueeze(0)
    return t, orig


def infer(model: nn.Module, tensor: torch.Tensor, thr: float = THRESHOLD):
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor.to(DEVICE))[0, 0]).cpu().numpy()
    return (prob >= thr).astype(np.uint8)


def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.42):
    f = rgb.astype(np.float32)
    r = np.zeros_like(f); r[:, :, 0] = 215.0
    m = np.stack([mask] * 3, -1).astype(np.float32)
    return (f * (1 - alpha * m) + r * (alpha * m)).clip(0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# Synthetic demo image
# ════════════════════════════════════════════════════════════════════════════

def make_demo_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    H, W = 400, 400
    cx = W // 2 + rng.integers(-30, 30)
    cy = H // 2 + rng.integers(-20, 20)

    # Skin background
    bg = np.stack([
        rng.normal(0.76, 0.04, (H, W)),
        rng.normal(0.58, 0.04, (H, W)),
        rng.normal(0.44, 0.04, (H, W)),
    ], axis=-1).clip(0, 1)

    Y, X = np.ogrid[:H, :W]
    vign = 1 - 0.35 * (((X - W/2)/(W/2))**2 + ((Y - H/2)/(H/2))**2)
    bg  *= vign[:, :, None]

    # Irregular lesion boundary
    rx, ry  = rng.integers(85, 115), rng.integers(70, 100)
    angles  = np.linspace(0, 2*np.pi, 360)
    r_noise = (1
               + 0.12 * np.sin(3*angles + rng.uniform(0, 2))
               + 0.07 * np.sin(7*angles + rng.uniform(0, 2))
               + 0.05 * np.sin(13*angles))
    dx, dy = (X - cx) / rx, (Y - cy) / ry
    base   = dx**2 + dy**2
    ang    = np.arctan2(Y - cy, X - cx)
    interp = np.interp(ang.ravel(), angles, r_noise).reshape(H, W)
    lesion_mask = (base < interp**2)
    lesion_soft = gaussian_filter(lesion_mask.astype(np.float32), sigma=3)

    # Lesion colour + pigment network
    lesion = np.stack([
        rng.normal(0.28, 0.06, (H, W)),
        rng.normal(0.14, 0.04, (H, W)),
        rng.normal(0.08, 0.03, (H, W)),
    ], axis=-1).clip(0, 1)
    grid  = (np.sin(X*0.25) * np.sin(Y*0.25))
    grid  = (grid - grid.min()) / (grid.max() - grid.min())
    lesion -= ((grid < 0.3).astype(np.float32) * 0.12)[:, :, None]
    lesion  = lesion.clip(0, 1)

    alpha = lesion_soft[:, :, None]
    img   = (1 - alpha) * bg + alpha * lesion

    # Hair artefacts
    for _ in range(rng.integers(4, 9)):
        x0 = rng.integers(0, W); y0 = rng.integers(0, H)
        a  = rng.uniform(0, np.pi); ln = rng.integers(60, 160)
        x1 = int(x0 + ln*np.cos(a)); y1 = int(y0 + ln*np.sin(a))
        tmp = (img*255).astype(np.uint8)
        cv2.line(tmp, (x0,y0), (x1,y1), (20,15,10), thickness=rng.integers(1,2))
        img = tmp.astype(np.float32)/255.0

    return Image.fromarray((img.clip(0,1)*255).astype(np.uint8))


# ════════════════════════════════════════════════════════════════════════════
# Grid figure
# ════════════════════════════════════════════════════════════════════════════

def build_grid(
    originals:    list,            # list of RGB uint8 np arrays
    predictions:  dict,            # {"ModelName": [binary_mask, ...]}
    show_overlay: bool  = False,
    dpi:          int   = DPI,
    save_path:    str   = OUTPUT_PATH,
):
    models   = list(predictions.keys())
    n_imgs   = len(originals)
    n_extra  = 2 if show_overlay else 1   # mask [+ overlay] per model
    n_cols   = 1 + len(models) * n_extra

    COL_W, ROW_H = 2.5, 2.5
    fig_w = COL_W * n_cols
    fig_h = ROW_H * n_imgs + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = GridSpec(n_imgs, n_cols, figure=fig,
                   hspace=0.05, wspace=0.03,
                   top=0.93, bottom=0.02, left=0.01, right=0.99)

    # ── Column headers ────────────────────────────────────────────────────────
    col_labels = ["Input Image"]
    for m in models:
        col_labels.append(f"{m}\nPrediction")
        if show_overlay:
            col_labels.append(f"{m}\nOverlay")

    header_y = 0.965
    for ci, lbl in enumerate(col_labels):
        x = (ci + 0.5) / n_cols
        fig.text(x, header_y, lbl, ha="center", va="top",
                 fontsize=9, fontweight="bold", color="#1a1a2e",
                 transform=fig.transFigure)

    # ── Thin horizontal line under headers ───────────────────────────────────
    fig.add_artist(plt.Line2D([0.01, 0.99], [0.945, 0.945],
                              transform=fig.transFigure,
                              color="#cccccc", linewidth=0.8))

    # ── Fill grid ─────────────────────────────────────────────────────────────
    for ri, orig in enumerate(originals):
        ci = 0

        def _ax(row, col, img, cmap=None, xlabel=None):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_edgecolor("#dddddd")
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=7.5, color="#555555",
                              labelpad=2)
            return ax

        # Input
        _ax(ri, ci, orig); ci += 1

        # Models
        for m in models:
            masks = predictions.get(m, [])
            if ri < len(masks) and masks[ri] is not None:
                binary = masks[ri]
                pct    = 100.0 * binary.sum() / binary.size
                _ax(ri, ci, binary, cmap="gray",
                    xlabel=f"lesion {pct:.1f}%"); ci += 1
                if show_overlay:
                    _ax(ri, ci, make_overlay(orig, binary)); ci += 1
            else:
                ax = fig.add_subplot(gs[ri, ci])
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="#999999", fontsize=9, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                ci += 1
                if show_overlay: ci += 1

    # ── Footer note ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.005,
             "ISIC Skin Lesion Segmentation — research use only",
             ha="center", fontsize=7, color="#aaaaaa",
             transform=fig.transFigure)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nGrid saved → {os.path.relpath(save_path, ROOT)}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def collect_images(args) -> list:
    images = []

    if args.images:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted([
            p for p in glob.glob(os.path.join(args.images, "*"))
            if os.path.splitext(p)[1].lower() in exts
        ])
        if not paths:
            print(f"[WARN] No images found in {args.images}. Using demo images.")
        else:
            paths = paths[:args.n]
            print(f"Loading {len(paths)} images from {args.images}")
            for p in paths:
                images.append((os.path.basename(p), Image.open(p).convert("RGB")))

    if not images:
        print(f"Generating {args.n} synthetic demo images (seeds {args.seed}–{args.seed+args.n-1})")
        for i in range(args.n):
            seed = args.seed + i
            images.append((f"demo_{seed}", make_demo_image(seed=seed)))

    return images


def main():
    parser = argparse.ArgumentParser(
        description="Test all models and save a prediction grid image.")
    parser.add_argument("--images",   default=None,
                        help="Folder with dermoscopic images (JPG/PNG). "
                             "If omitted, synthetic demo images are used.")
    parser.add_argument("--n",        type=int, default=4,
                        help="Number of images to include in the grid (default 4).")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Seed for the first demo image (default 0).")
    parser.add_argument("--threshold",type=float, default=THRESHOLD,
                        help=f"Segmentation threshold (default {THRESHOLD}).")
    parser.add_argument("--overlay",  action="store_true",
                        help="Add overlay columns alongside mask columns.")
    parser.add_argument("--dpi",      type=int, default=DPI,
                        help=f"Output image DPI (default {DPI}).")
    parser.add_argument("--output",   default=OUTPUT_PATH,
                        help="Output PNG path.")
    parser.add_argument("--models",   nargs="+",
                        default=["U-Net", "SegNet", "DeepLabV3"],
                        choices=["U-Net", "SegNet", "DeepLabV3"],
                        help="Which models to run (default: all three).")
    args = parser.parse_args()

    print(f"\nISIC Test Grid")
    print(f"  Device    : {DEVICE}")
    print(f"  Models    : {', '.join(args.models)}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Overlay   : {args.overlay}")
    print(f"  DPI       : {args.dpi}\n")

    # ── Collect images ────────────────────────────────────────────────────────
    image_list = collect_images(args)   # [(name, PIL), ...]

    # ── Load models & run inference ───────────────────────────────────────────
    predictions = {}   # {model_name: [binary_mask, ...]}
    originals   = []   # RGB uint8 arrays for the grid

    preprocessed = []
    for name, pil in image_list:
        t, orig = preprocess(pil)
        preprocessed.append(t)
        originals.append(orig)

    for model_name in args.models:
        ckpt_path = CHECKPOINTS.get(model_name)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            print(f"  [SKIP] {model_name}: checkpoint not found at "
                  f"{os.path.relpath(ckpt_path or '?', ROOT)}")
            predictions[model_name] = [None] * len(originals)
            continue

        try:
            model = load_model(model_name, ckpt_path)
            masks = []
            for i, tensor in enumerate(preprocessed):
                binary = infer(model, tensor, args.threshold)
                masks.append(binary)
                pct = 100.0 * binary.sum() / binary.size
                name_img = image_list[i][0]
                print(f"    {model_name} | {name_img:30s} | lesion {pct:5.1f}%")
            predictions[model_name] = masks
        except Exception as exc:
            print(f"  [ERROR] {model_name}: {exc}")
            predictions[model_name] = [None] * len(originals)

    # ── Build and save grid ───────────────────────────────────────────────────
    print("\nBuilding grid figure…")
    build_grid(
        originals    = originals,
        predictions  = predictions,
        show_overlay = args.overlay,
        dpi          = args.dpi,
        save_path    = args.output,
    )


if __name__ == "__main__":
    main()
