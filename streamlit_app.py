"""
ISIC Skin Lesion Segmentation System — Streamlit Application
"""

import os, sys, json, time, warnings, tempfile
from io import BytesIO

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models.segmentation as seg_models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from src.utils import dice_score, iou_score  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEFAULT_CHECKPOINTS = {
    "U-Net":     os.path.join(BASE_DIR, "models",   "best_unet_isic.pth"),
    "SegNet":    os.path.join(BASE_DIR, "outputs",  "segnet",    "best_segnet.pth"),
    "DeepLabV3": os.path.join(BASE_DIR, "outputs",  "deeplabv3", "best_deeplabv3_isic.pth"),
}
METRICS_FILES = {
    "U-Net":     os.path.join(BASE_DIR, "outputs", "test_metrics.json"),
    "SegNet":    os.path.join(BASE_DIR, "outputs", "segnet",    "conclusion_segnet.json"),
    "DeepLabV3": os.path.join(BASE_DIR, "outputs", "deeplabv3", "test_metrics.json"),
}
HISTORY_FILES = {
    "SegNet":    os.path.join(BASE_DIR, "outputs", "segnet",    "metrics_segnet.json"),
    "DeepLabV3": os.path.join(BASE_DIR, "outputs", "deeplabv3", "test_metrics.json"),
}
CONFUSION_DIR = os.path.join(BASE_DIR, "outputs", "confusion_matrix")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hugging Face auto-download ─────────────────────────────────────────────────
# Set HF_REPO to your Hugging Face model repo, e.g. "yourname/isic-models"
# Files in that repo must be named exactly as in HF_MODEL_FILES below.
# Leave HF_REPO = "" to disable auto-download (models must be present locally).
HF_REPO = os.environ.get("HF_REPO", "")   # set via Streamlit secrets or env var

HF_MODEL_FILES = {
    "U-Net":     ("models/best_unet_isic.pth",              "best_unet_isic.pth"),
    "SegNet":    ("outputs/segnet/best_segnet.pth",          "best_segnet.pth"),
    "DeepLabV3": ("outputs/deeplabv3/best_deeplabv3_isic.pth","best_deeplabv3_isic.pth"),
}

@st.cache_resource(show_spinner=False)
def _download_checkpoints():
    if not HF_REPO:
        return
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN", None)
        for name, (local_rel, hf_file) in HF_MODEL_FILES.items():
            local_abs = os.path.join(BASE_DIR, local_rel)
            if os.path.isfile(local_abs):
                continue
            os.makedirs(os.path.dirname(local_abs), exist_ok=True)
            with st.spinner(f"Downloading {name} from Hugging Face…"):
                hf_hub_download(repo_id=HF_REPO, filename=hf_file,
                                local_dir=os.path.dirname(local_abs), token=token)
    except Exception as e:
        st.warning(f"Model download skipped: {e}")

BENCHMARK = {
    "U-Net":     {"Dice": 0.8664, "IoU": 0.7899, "Precision": 0.9014, "Recall": 0.8802,
                  "Params": "31.0 M", "Epochs": 20},
    "SegNet":    {"Dice": 0.8575, "IoU": 0.7762, "Precision": 0.8815, "Recall": 0.8832,
                  "Params": "24.9 M", "Epochs": 30},
    "DeepLabV3": {"Dice": 0.9030, "IoU": 0.8263, "Precision": 0.9251, "Recall": 0.9186,
                  "Params": "39.6 M", "Epochs": 47},
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISIC Skin Lesion Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
_download_checkpoints()   # no-op locally; auto-downloads on Streamlit Cloud if HF_REPO is set

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 4rem;
    max-width: 1300px;
}

/* ── Sidebar shell ───────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.08);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Sidebar logo header ─────────────────────────────────── */
.sb-header {
    padding: 1.6rem 1.2rem 1.1rem;
    border-bottom: 1px solid rgba(128,128,128,0.1);
    margin-bottom: 0.5rem;
}
.sb-icon-wrap {
    width: 44px; height: 44px;
    background: var(--primary-color);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    margin-bottom: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15);
}
.sb-title {
    font-size: 0.95rem; font-weight: 800;
    color: var(--text-color);
    letter-spacing: -0.02em; line-height: 1.2;
}
.sb-sub {
    font-size: 0.66rem; color: var(--text-color);
    opacity: 0.38; margin-top: 4px; letter-spacing: 0.02em;
}

/* ── Nav label ───────────────────────────────────────────── */
.sb-nav-label {
    font-size: 0.6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: var(--text-color); opacity: 0.35;
    padding: 0.9rem 1.2rem 0.4rem;
}

/* ── Transform st.radio → nav pills ─────────────────────── */
[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
    gap: 2px !important;
    padding: 0 0.6rem;
}

/* Hide the radio circle dot */
[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* Each nav item */
[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    border-radius: 10px !important;
    padding: 9px 14px !important;
    margin: 1px 0 !important;
    cursor: pointer !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    transition: background 0.15s ease !important;
}

/* Nav item text */
[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: var(--text-color) !important;
    opacity: 0.65 !important;
}

/* Hover */
[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background: rgba(128,128,128,0.08) !important;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
    opacity: 0.9 !important;
}

/* Active nav item — targets checked radio */
[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
    background: var(--primary-color) !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.15) !important;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
    color: white !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* ── Device info footer ──────────────────────────────────── */
.sb-footer {
    padding: 0.9rem 1.2rem;
    border-top: 1px solid rgba(128,128,128,0.08);
    margin-top: 1rem;
}
.sb-footer-row {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.69rem; color: var(--text-color);
    opacity: 0.4; margin-bottom: 5px;
}
.sb-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e; flex-shrink: 0;
    box-shadow: 0 0 5px #22c55e;
}

/* ── Page hero ───────────────────────────────────────────── */
.hero-banner {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.1);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
}
.hero-banner .hero-tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(128,128,128,0.1);
    color: var(--text-color);
    opacity: 0.7;
    margin-bottom: 12px;
}
.hero-banner .hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text-color);
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 10px;
}
.hero-banner .hero-sub {
    font-size: 0.92rem;
    color: var(--text-color);
    opacity: 0.6;
    max-width: 580px;
    line-height: 1.6;
    margin: 0;
}

/* ── Section label ───────────────────────────────────────── */
.sec-label {
    font-size: 0.64rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-color);
    opacity: 0.5;
    margin: 2rem 0 0.8rem;
    padding-left: 10px;
    border-left: 3px solid var(--primary-color);
}

/* ── Stat card ───────────────────────────────────────────── */
.stat-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.1);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(0,0,0,0.09);
}
.stat-card .s-val {
    font-size: 1.95rem;
    font-weight: 800;
    color: var(--primary-color);
    line-height: 1.05;
    display: block;
    letter-spacing: -0.03em;
}
.stat-card .s-val.best { color: #16a34a; }
.stat-card .s-lbl {
    font-size: 0.64rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-color);
    opacity: 0.48;
    margin-top: 6px;
    display: block;
}

/* ── Image caption ───────────────────────────────────────── */
.img-cap {
    text-align: center;
    font-size: 0.67rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-color);
    opacity: 0.42;
    margin-top: 6px;
}

/* ── Model card ──────────────────────────────────────────── */
.model-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.1);
    border-left: 4px solid var(--primary-color);
    border-radius: 0 14px 14px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.model-card:hover {
    transform: translateX(4px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.08);
}
.model-card .m-name {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text-color);
    letter-spacing: -0.01em;
}
.model-card .m-desc {
    font-size: 0.79rem;
    color: var(--text-color);
    opacity: 0.58;
    margin-top: 4px;
    line-height: 1.45;
}
.model-card .m-score {
    font-size: 0.79rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-top: 8px;
}
.model-card .m-arch {
    font-size: 0.69rem;
    color: var(--text-color);
    opacity: 0.38;
    margin-top: 2px;
}

/* ── Info row ────────────────────────────────────────────── */
.info-row {
    display: flex;
    border-bottom: 1px solid rgba(128,128,128,0.09);
    padding: 7px 4px;
    font-size: 0.83rem;
    transition: background 0.1s;
}
.info-row:hover { background: rgba(128,128,128,0.04); border-radius: 6px; }
.info-row .ik {
    width: 44%;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.65;
}
.info-row .iv {
    color: var(--text-color);
    opacity: 0.9;
    font-weight: 500;
}

/* ── Disclaimer ──────────────────────────────────────────── */
.disclaimer {
    font-size: 0.69rem;
    color: var(--text-color);
    opacity: 0.32;
    border-top: 1px solid rgba(128,128,128,0.1);
    padding-top: 1.5rem;
    margin-top: 4rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sec(title: str):
    st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)

def img_cap(text: str):
    st.markdown(f'<div class="img-cap">{text}</div>', unsafe_allow_html=True)

def stat_card(val: str, lbl: str, best: bool = False):
    cls = "s-val best" if best else "s-val"
    return (f'<div class="stat-card"><span class="{cls}">{val}</span>'
            f'<span class="s-lbl">{lbl}</span></div>')

def footer():
    st.markdown(
        '<div class="disclaimer">This tool is for research and educational '
        'purposes only and is not intended for clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )

def info_row(key: str, val: str):
    return (f'<div class="info-row"><span class="ik">{key}</span>'
            f'<span class="iv">{val}</span></div>')

def pil_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = BytesIO(); img.save(buf, format=fmt); return buf.getvalue()


# ── Model architectures (must match trained checkpoint keys) ──────────────────

class _DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
        )
    def forward(self, x): return self.conv(x)

class _UNet(nn.Module):
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

class _SegNet(nn.Module):
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

class _DeepLabV3(nn.Module):
    def __init__(self, out_channels=1, pretrained=False):
        super().__init__()
        weights = seg_models.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        # aux_loss=True: checkpoint was always saved with the aux branch present
        self.model = seg_models.deeplabv3_resnet50(weights=weights, aux_loss=True)
        self.model.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)
    def forward(self, x):
        out = self.model(x)["out"]
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


# ── Preprocessing ─────────────────────────────────────────────────────────────
_tf = A.Compose([A.Resize(IMG_SIZE,IMG_SIZE),
                 A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD), ToTensorV2()])

def preprocess(pil: Image.Image):
    arr   = np.array(pil.convert("RGB"))
    orig  = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t     = _tf(image=arr)["image"].unsqueeze(0)
    return t, orig

def overlay(rgb: np.ndarray, mask: np.ndarray, alpha=0.42):
    f = rgb.astype(np.float32)
    r = np.zeros_like(f); r[:,:,0] = 215.0
    m = np.stack([mask]*3, -1).astype(np.float32)
    return (f*(1-alpha*m) + r*(alpha*m)).clip(0,255).astype(np.uint8)

def mask_pil(m: np.ndarray): return Image.fromarray((m*255).astype(np.uint8), "L")

def heatmap_pil(prob: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3,3), dpi=96)
    ax.imshow(prob, cmap="magma", vmin=0, vmax=1); ax.axis("off")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout(pad=0.2)
    buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()


# ── Demo image ────────────────────────────────────────────────────────────────

def make_demo_image(seed: int = 0) -> Image.Image:
    """
    Synthetic dermoscopic-like image for testing without real data.
    Simulates: skin-tone background + dark oval lesion + pigment network texture.
    """
    rng = np.random.default_rng(seed)
    H, W = 400, 400
    cx, cy = W // 2 + rng.integers(-30, 30), H // 2 + rng.integers(-20, 20)

    # ── Skin background ───────────────────────────────────────────────────────
    bg = np.stack([
        rng.normal(0.76, 0.04, (H, W)),   # R
        rng.normal(0.58, 0.04, (H, W)),   # G
        rng.normal(0.44, 0.04, (H, W)),   # B
    ], axis=-1).clip(0, 1)

    # Vignette (dark corners like real dermoscope)
    Y, X = np.ogrid[:H, :W]
    vign  = 1 - 0.35 * (((X - W/2)/(W/2))**2 + ((Y - H/2)/(H/2))**2)
    bg   *= vign[:, :, None]

    # ── Lesion mask (irregular ellipse via polar distortion) ──────────────────
    rx = rng.integers(85, 115)
    ry = rng.integers(70, 100)
    angles = np.linspace(0, 2*np.pi, 360)
    # irregular radius: sum of sinusoids
    r_noise = (1
               + 0.12 * np.sin(3 * angles + rng.uniform(0, 2))
               + 0.07 * np.sin(7 * angles + rng.uniform(0, 2))
               + 0.05 * np.sin(13 * angles))

    # Build mask from the irregular ellipse
    dx, dy = (X - cx) / rx, (Y - cy) / ry
    base   = dx**2 + dy**2
    ang    = np.arctan2(Y - cy, X - cx)
    interp = np.interp(ang.ravel(), angles, r_noise).reshape(H, W)
    lesion_mask = (base < interp**2)

    # Gaussian smooth the boundary
    from scipy.ndimage import gaussian_filter
    lesion_soft = gaussian_filter(lesion_mask.astype(np.float32), sigma=3)

    # ── Lesion colour (dark brown with internal variation) ────────────────────
    lesion_base = np.stack([
        rng.normal(0.28, 0.06, (H, W)),   # R
        rng.normal(0.14, 0.04, (H, W)),   # G
        rng.normal(0.08, 0.03, (H, W)),   # B
    ], axis=-1).clip(0, 1)

    # Pigment network: fine dark grid pattern inside lesion
    grid  = (np.sin(X * 0.25) * np.sin(Y * 0.25))
    grid  = (grid - grid.min()) / (grid.max() - grid.min())
    net   = (grid < 0.3).astype(np.float32) * 0.12
    lesion_base -= net[:, :, None]
    lesion_base  = lesion_base.clip(0, 1)

    # ── Blend skin + lesion ────────────────────────────────────────────────────
    alpha = lesion_soft[:, :, None]
    img   = (1 - alpha) * bg + alpha * lesion_base

    # Hair artefacts (thin dark lines)
    for _ in range(rng.integers(4, 9)):
        x0 = rng.integers(0, W); y0 = rng.integers(0, H)
        angle = rng.uniform(0, np.pi)
        length = rng.integers(60, 160)
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        tmp = (img * 255).astype(np.uint8)
        cv2.line(tmp, (x0, y0), (x1, y1), (20, 15, 10),
                 thickness=rng.integers(1, 2))
        img = tmp.astype(np.float32) / 255.0

    return Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8))


# ── Demo gallery helpers ───────────────────────────────────────────────────────

DEMO_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)

@st.cache_data(show_spinner=False)
def _prerender_thumbs(seeds: tuple, size: int = 130) -> list:
    out = []
    for s in seeds:
        pil = make_demo_image(seed=s).resize((size, size), Image.LANCZOS)
        buf = BytesIO(); pil.save(buf, format="PNG"); buf.seek(0)
        out.append(buf.getvalue())
    return out

def _demo_selector(prefix: str) -> bool:
    """8-image thumbnail gallery. Returns True when a new image is selected."""
    thumbs   = _prerender_thumbs(DEMO_SEEDS)
    selected = st.session_state.get(f"{prefix}_demo_seed", None)
    triggered = False

    row1, row2 = st.columns(4, gap="small"), st.columns(4, gap="small")
    for i, (seed, tbytes) in enumerate(zip(DEMO_SEEDS, thumbs)):
        col = (row1 if i < 4 else row2)[i % 4]
        with col:
            is_sel = (selected == seed)
            border = ("2px solid var(--primary-color)" if is_sel
                      else "2px solid rgba(128,128,128,0.2)")
            st.markdown(
                f'<div style="border:{border};border-radius:8px;overflow:hidden;margin-bottom:4px">'
                f'</div>', unsafe_allow_html=True)
            st.image(Image.open(BytesIO(tbytes)), use_container_width=True)
            lbl = "Selected" if is_sel else f"Seed {seed}"
            if st.button(lbl, key=f"{prefix}_demo_{seed}",
                         type="primary" if is_sel else "secondary",
                         use_container_width=True):
                demo_pil = make_demo_image(seed=seed)
                buf = BytesIO(); demo_pil.save(buf, format="PNG"); buf.seek(0)
                st.session_state[f"{prefix}_demo_bytes"] = buf.getvalue()
                st.session_state[f"{prefix}_demo_seed"]  = seed
                st.session_state[f"{prefix}_img_name"]   = f"__demo_{seed}__"
                st.session_state.pop(f"{prefix}_res",  None)
                st.session_state.pop(f"{prefix}_orig", None)
                triggered = True
    return triggered


# ── Model loading ─────────────────────────────────────────────────────────────

def _parse_ckpt(path):
    c = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(c, dict) and "model_state_dict" in c:
        return c["model_state_dict"], {k:v for k,v in c.items() if k!="model_state_dict"}
    return c, {}

@st.cache_resource(show_spinner=False)
def load_model(name: str, ckpt_path: str):
    m = {"U-Net": _UNet, "SegNet": _SegNet}
    if name in m:
        model = m[name]()
    elif name == "DeepLabV3":
        model = _DeepLabV3(pretrained=False)
    else:
        raise ValueError(name)
    sd, meta = _parse_ckpt(ckpt_path)
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval(), meta

def infer(model, tensor, thr=0.5):
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor.to(DEVICE))[0,0]).cpu().numpy().astype(np.float32)
    return prob, (prob >= thr).astype(np.uint8)


# ── Metrics ───────────────────────────────────────────────────────────────────

def full_metrics(pred_bin: np.ndarray, gt_pil: Image.Image):
    gt  = (np.array(gt_pil.convert("L").resize((IMG_SIZE,IMG_SIZE), Image.NEAREST)) > 127).astype(np.float32)
    p   = pred_bin.astype(np.float32); e = 1e-6
    tp  = (p*gt).sum(); fp=(p*(1-gt)).sum(); fn=((1-p)*gt).sum()
    return {
        "Dice":      float((2*tp+e)/(2*tp+fp+fn+e)),
        "IoU":       float((tp+e)/(tp+fp+fn+e)),
        "Precision": float((tp+e)/(tp+fp+e)),
        "Recall":    float((tp+e)/(tp+fn+e)),
    }

def load_history(name: str) -> dict:
    p = HISTORY_FILES.get(name, "")
    if not os.path.isfile(p): return {}
    raw = json.load(open(p))
    h   = raw.get("history", raw)
    return {k: h[k] for k in ("train_loss","val_loss","train_dice","val_dice") if k in h}


# ── Checkpoint selector widget ─────────────────────────────────────────────────

def ckpt_selector(name: str, prefix: str):
    src = st.radio("Source", ["Default path","Upload .pth"], key=f"{prefix}_src",
                   horizontal=True)
    if src == "Default path":
        p = DEFAULT_CHECKPOINTS[name]
        ok = os.path.isfile(p)
        st.caption(f"`{os.path.relpath(p, BASE_DIR)}`")
        st.success("Checkpoint found", icon=None) if ok else st.error("File not found")
        return p if ok else None
    up = st.file_uploader("Select file", type=["pth","pt"], key=f"{prefix}_up",
                          label_visibility="collapsed")
    if up:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        t.write(up.read()); t.flush(); return t.name
    return None


# ── Training curves plot ───────────────────────────────────────────────────────

def build_curve_fig(histories: dict, dark: bool) -> plt.Figure:
    bg   = "#0e1117" if dark else "#ffffff"
    fg   = "#fafafa" if dark else "#111111"
    grid = "#2a2a2a" if dark else "#e5e7eb"
    palette = {"SegNet": "#60a5fa", "DeepLabV3": "#34d399"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=bg)
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=8)
        ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values(): spine.set_edgecolor(grid)
        ax.grid(True, color=grid, linewidth=0.6, alpha=0.7)
        ax.set_xlabel("Epoch", fontsize=9)

    for name, h in histories.items():
        c = palette.get(name, "#a78bfa")
        if "val_loss"  in h: axes[0].plot(h["val_loss"],  color=c, lw=2,   label=f"{name} val")
        if "train_loss"in h: axes[0].plot(h["train_loss"],color=c, lw=1.1, ls="--", alpha=0.5)
        if "val_dice"  in h: axes[1].plot(h["val_dice"],  color=c, lw=2,   label=f"{name} val")
        if "train_dice"in h: axes[1].plot(h["train_dice"],color=c, lw=1.1, ls="--", alpha=0.5)

    axes[0].set_title("Loss",            fontsize=10, fontweight="bold")
    axes[1].set_title("Dice Coefficient",fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Loss",  fontsize=9); axes[1].set_ylabel("Dice", fontsize=9)
    axes[1].set_ylim(0.70, 1.00)
    for ax in axes:
        leg = ax.legend(fontsize=8, framealpha=0.15)
        for t in leg.get_texts(): t.set_color(fg)
    fig.suptitle("Solid = Validation   ·   Dashed = Training", fontsize=8,
                 color=fg, alpha=0.55, y=0.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

_NAV_ICONS = {
    "Home":       "🏠",
    "Prediction": "⚡",
    "Comparison": "⚖",
    "Metrics":    "📊",
}

def sidebar() -> str:
    with st.sidebar:
        # ── Header ────────────────────────────────────────────
        st.markdown("""
        <div class="sb-header">
            <div class="sb-icon-wrap">🔬</div>
            <div class="sb-title">ISIC Segmentation</div>
            <div class="sb-sub">ISIC 2018 &nbsp;·&nbsp; Task 1 &nbsp;·&nbsp; Deep Learning</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Nav label ─────────────────────────────────────────
        st.markdown('<div class="sb-nav-label">Navigation</div>',
                    unsafe_allow_html=True)

        page = st.radio("nav", list(_NAV_ICONS.keys()),
                        label_visibility="collapsed")

        # ── Footer ────────────────────────────────────────────
        dev  = ("GPU — " + torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU")
        gpu  = torch.cuda.is_available()
        st.markdown(
            f'<div class="sb-footer">'
            f'<div class="sb-footer-row">'
            f'<div class="sb-dot" style="background:{"#22c55e" if gpu else "#94a3b8"};'
            f'box-shadow:{"0 0 5px #22c55e" if gpu else "none"}"></div>'
            f'{dev}</div>'
            f'<div class="sb-footer-row" style="margin-bottom:0">'
            f'PyTorch&nbsp;{torch.__version__}</div>'
            f'</div>',
            unsafe_allow_html=True)

    return page


# ═════════════════════════════════════════════════════════════════════════════
# Page — Home
# ═════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-tag">Research Project &nbsp;·&nbsp; ISIC 2018 Task 1</div>
        <div class="hero-title">Skin Lesion Segmentation</div>
        <p class="hero-sub">
            Automatic pixel-wise segmentation of dermoscopic images using three deep learning
            architectures trained on the ISIC 2018 benchmark dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        sec("System overview")
        st.markdown(
            "Three segmentation architectures are evaluated on **390 held-out test images** "
            "from the ISIC 2018 dataset. All models are trained end-to-end with a combined "
            "**Dice + Binary Cross-Entropy loss** and evaluated with Dice coefficient and IoU."
        )

        sec("Benchmark — Test set (390 images · 256 × 256 px)")
        best_dice = max(v["Dice"] for v in BENCHMARK.values())
        best_iou  = max(v["IoU"]  for v in BENCHMARK.values())

        df_data = {
            "Model":     list(BENCHMARK.keys()),
            "Dice":      [v["Dice"]      for v in BENCHMARK.values()],
            "IoU":       [v["IoU"]       for v in BENCHMARK.values()],
            "Precision": [v["Precision"] for v in BENCHMARK.values()],
            "Recall":    [v["Recall"]    for v in BENCHMARK.values()],
            "Params":    [v["Params"]    for v in BENCHMARK.values()],
            "Epochs":    [v["Epochs"]    for v in BENCHMARK.values()],
        }
        df = pd.DataFrame(df_data).set_index("Model")

        def _style(v, col):
            if col in ("Dice","IoU","Precision","Recall"):
                best = best_dice if col=="Dice" else (best_iou if col=="IoU" else None)
                if best and abs(float(v)-best) < 1e-4:
                    return "font-weight:700; color:#16a34a"
            return ""

        styled = df.style.format({"Dice":"{:.4f}","IoU":"{:.4f}",
                                   "Precision":"{:.4f}","Recall":"{:.4f}"})
        st.dataframe(styled, use_container_width=True)
        st.caption("Best value per metric highlighted in the Metrics page.")

        sec("Dataset")
        rows = "".join([
            info_row("Dataset",         "ISIC 2018 — Skin Lesion Analysis"),
            info_row("Task",            "Task 1 — Lesion Boundary Segmentation"),
            info_row("Total images",    "2,594  (train 1,815 · val 389 · test 390)"),
            info_row("Modality",        "Dermoscopy (epiluminescence microscopy)"),
            info_row("Input resolution","256 × 256 pixels"),
            info_row("Normalisation",   "ImageNet mean and standard deviation"),
            info_row("Loss function",   "Dice Loss + Binary Cross-Entropy"),
        ])
        st.markdown(f'<div style="margin-top:4px">{rows}</div>', unsafe_allow_html=True)

    with col_right:
        sec("Models")

        descs = {
            "U-Net": ("Encoder-decoder with symmetric skip connections. "
                      "Reference standard for medical image segmentation since 2015.",
                      "Features: 64→128→256→512"),
            "SegNet": ("VGG-style encoder with index-based max-unpooling. "
                       "Retains precise spatial boundaries through pooling indices.",
                       "5-stage encoder / decoder"),
            "DeepLabV3": ("ResNet-50 backbone with Atrous Spatial Pyramid Pooling. "
                          "Best performer on this benchmark.",
                          "Dilated convolutions · ASPP module"),
        }
        for name, (desc, arch) in descs.items():
            b = BENCHMARK[name]
            st.markdown(
                f'<div class="model-card">'
                f'<div class="m-name">{name}</div>'
                f'<div class="m-desc">{desc}</div>'
                f'<div class="m-score">Dice {b["Dice"]:.4f} · IoU {b["IoU"]:.4f}</div>'
                f'<div class="m-arch">{arch} · {b["Params"]} params · {b["Epochs"]} epochs</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        sec("System")
        dev = ("GPU — "+torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
        rows2 = "".join([
            info_row("Compute device", dev),
            info_row("Framework",      f"PyTorch {torch.__version__}"),
            info_row("Input size",     "256 × 256 px"),
        ])
        st.markdown(f'<div style="margin-top:4px">{rows2}</div>', unsafe_allow_html=True)

    footer()


# ═════════════════════════════════════════════════════════════════════════════
# Page — Prediction
# ═════════════════════════════════════════════════════════════════════════════

def page_prediction():
    st.title("Prediction")
    st.markdown("Run segmentation inference on a single dermoscopic image.")
    st.divider()

    with st.sidebar:
        st.markdown("### Model")
        model_name = st.selectbox("Architecture", ["U-Net","SegNet","DeepLabV3"], key="p_model")
        ckpt_path  = ckpt_selector(model_name, "p")
        st.markdown("---")
        st.markdown("### Settings")
        thr          = st.slider("Threshold", 0.10, 0.90, 0.50, 0.05, key="p_thr")
        show_heatmap = st.toggle("Show probability heatmap", False, key="p_hm")
        st.markdown("---")
        st.markdown("### Ground Truth (optional)")
        st.caption("Upload a binary mask to compute metrics.")
        gt_file = st.file_uploader("Mask", type=["png","jpg","jpeg"],
                                   key="p_gt", label_visibility="collapsed")

    # ── Image source ──────────────────────────────────────────────────────────
    tab_demo, tab_upload = st.tabs(["Demo Gallery", "Upload Image"])
    auto_run = False
    up_file  = None

    with tab_demo:
        st.caption("Click any thumbnail — inference runs automatically.")
        auto_run = _demo_selector("p")

    with tab_upload:
        up_file = st.file_uploader("Dermoscopic image (JPG / PNG)",
                                   type=["jpg","jpeg","png"], key="p_img",
                                   label_visibility="collapsed")
        if up_file and st.session_state.get("p_img_name") != up_file.name:
            st.session_state.pop("p_res",       None)
            st.session_state.pop("p_demo_bytes",None)
            st.session_state.pop("p_demo_seed", None)
            st.session_state["p_img_name"] = up_file.name

    # Resolve active image (upload takes priority)
    active_pil: Image.Image | None = None
    if up_file:
        active_pil = Image.open(up_file).convert("RGB")
    elif "p_demo_bytes" in st.session_state:
        active_pil = Image.open(BytesIO(st.session_state["p_demo_bytes"])).convert("RGB")

    if not active_pil:
        st.info("Choose a demo image above or upload your own to begin.")
        footer(); return

    if not ckpt_path:
        st.warning("Select or upload a valid checkpoint to proceed.")
        footer(); return

    # Auto-run on demo selection; manual button for uploads
    run_btn = st.button("Run Inference", type="primary")
    should_run = run_btn or (auto_run and not up_file)

    if should_run:
        with st.spinner(f"Running {model_name}…"):
            try:
                model, meta  = load_model(model_name, ckpt_path)
                tensor, orig = preprocess(active_pil)
                t0           = time.perf_counter()
                prob, binary = infer(model, tensor, thr)
                elapsed      = time.perf_counter() - t0
                st.session_state["p_res"] = dict(
                    orig=orig, prob=prob, binary=binary,
                    elapsed=elapsed, meta=meta, model=model_name)
            except Exception as e:
                st.error(f"Inference error: {e}"); footer(); return

    if "p_res" not in st.session_state:
        footer(); return

    res    = st.session_state["p_res"]
    orig   = res["orig"]; prob = res["prob"]; binary = res["binary"]
    ov     = overlay(orig, binary)
    meta   = res["meta"]

    # ── Images ────────────────────────────────────────────────────────────────
    sec("Segmentation Results")
    n    = 4 if show_heatmap else 3
    cols = st.columns(n, gap="small")
    with cols[0]: st.image(orig,               use_container_width=True); img_cap("Input")
    with cols[1]: st.image(mask_pil(binary),   use_container_width=True); img_cap("Predicted Mask")
    with cols[2]: st.image(Image.fromarray(ov),use_container_width=True); img_cap("Overlay")
    if show_heatmap:
        with cols[3]: st.image(heatmap_pil(prob), use_container_width=True); img_cap("Probability Map")

    # ── Stats ─────────────────────────────────────────────────────────────────
    sec("Inference Statistics")
    pct = 100.0 * binary.sum() / binary.size
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Inference time", f"{res['elapsed']*1000:.1f} ms")
    c2.metric("Lesion area",    f"{pct:.1f}%")
    c3.metric("Lesion pixels",  f"{int(binary.sum()):,}")
    c4.metric("Max confidence", f"{prob.max():.3f}")

    bar_w = min(int(pct), 100)
    st.markdown(
        f'<div style="margin-top:6px">'
        f'<div style="font-size:.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.1em;opacity:.45;margin-bottom:5px">Lesion Coverage</div>'
        f'<div style="background:var(--secondary-background-color);border-radius:20px;'
        f'height:10px;overflow:hidden;border:1px solid rgba(128,128,128,.15)">'
        f'<div style="background:var(--primary-color);width:{bar_w}%;height:100%;'
        f'border-radius:20px"></div></div>'
        f'<div style="font-size:.7rem;opacity:.4;margin-top:3px">'
        f'{pct:.2f}% of image area classified as lesion</div></div>',
        unsafe_allow_html=True)

    # ── GT metrics ────────────────────────────────────────────────────────────
    if gt_file:
        m = full_metrics(binary, Image.open(gt_file))
        sec("Evaluation vs Ground Truth")
        mc1, mc2, mc3, mc4 = st.columns(4, gap="small")
        mc1.metric("Dice",      f"{m['Dice']:.4f}")
        mc2.metric("IoU",       f"{m['IoU']:.4f}")
        mc3.metric("Precision", f"{m['Precision']:.4f}")
        mc4.metric("Recall",    f"{m['Recall']:.4f}")

    # ── Checkpoint info ───────────────────────────────────────────────────────
    if meta:
        with st.expander("Checkpoint details"):
            cols_m = st.columns(3, gap="small")
            if "epoch"    in meta: cols_m[0].metric("Epoch",    meta["epoch"])
            if "val_dice" in meta: cols_m[1].metric("Val Dice", f"{meta['val_dice']:.4f}")
            if "val_iou"  in meta: cols_m[2].metric("Val IoU",  f"{meta['val_iou']:.4f}")

    # ── Downloads ─────────────────────────────────────────────────────────────
    sec("Export")
    dc1, dc2 = st.columns(2, gap="small")
    dc1.download_button("Download Mask",    pil_bytes(mask_pil(binary)),
                        "mask.png","image/png", use_container_width=True)
    dc2.download_button("Download Overlay", pil_bytes(Image.fromarray(ov)),
                        "overlay.png","image/png", use_container_width=True)

    footer()


# ═════════════════════════════════════════════════════════════════════════════
# Page — Comparison
# ═════════════════════════════════════════════════════════════════════════════

def page_comparison():
    st.title("Model Comparison")
    st.markdown("Compare all three models on a single image.")
    st.divider()

    with st.sidebar:
        st.markdown("### Checkpoints")
        ckpt_paths = {}
        for name in ["U-Net","SegNet","DeepLabV3"]:
            p, ok = DEFAULT_CHECKPOINTS[name], os.path.isfile(DEFAULT_CHECKPOINTS[name])
            if ok: ckpt_paths[name] = p
            status = "found" if ok else "missing — upload below"
            with st.expander(f"{name}  [{status}]", expanded=not ok):
                st.caption(f"`{os.path.relpath(p, BASE_DIR)}`")
                if not ok:
                    up = st.file_uploader(f"{name} checkpoint",
                                          type=["pth","pt"], key=f"c_{name}",
                                          label_visibility="collapsed")
                    if up:
                        t = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
                        t.write(up.read()); t.flush(); ckpt_paths[name] = t.name
        st.markdown("---")
        thr = st.slider("Threshold", 0.10, 0.90, 0.50, 0.05, key="c_thr")

    # ── Image source ──────────────────────────────────────────────────────────
    tab_demo, tab_upload = st.tabs(["Demo Gallery", "Upload Image"])
    auto_run = False
    up_file  = None

    with tab_demo:
        st.caption("Click any thumbnail — all three models run automatically.")
        auto_run = _demo_selector("c")

    with tab_upload:
        up_file = st.file_uploader("Dermoscopic image (JPG / PNG)",
                                   type=["jpg","jpeg","png"], key="c_img",
                                   label_visibility="collapsed")
        if up_file and st.session_state.get("c_img_name") != up_file.name:
            st.session_state.pop("c_res",       None)
            st.session_state.pop("c_orig",      None)
            st.session_state.pop("c_demo_bytes",None)
            st.session_state.pop("c_demo_seed", None)
            st.session_state["c_img_name"] = up_file.name

    # Resolve active image (upload takes priority)
    active_pil: Image.Image | None = None
    if up_file:
        active_pil = Image.open(up_file).convert("RGB")
    elif "c_demo_bytes" in st.session_state:
        active_pil = Image.open(BytesIO(st.session_state["c_demo_bytes"])).convert("RGB")

    if not active_pil:
        st.info("Choose a demo image above or upload your own.")
        footer(); return

    available = {k: v for k, v in ckpt_paths.items() if os.path.isfile(v)}
    if not available:
        st.warning("No valid checkpoints available."); footer(); return

    run_btn    = st.button("Run All Models", type="primary")
    should_run = run_btn or (auto_run and not up_file)

    if should_run:
        tensor, orig = preprocess(active_pil)
        results, bar = {}, st.progress(0, text="Starting…")
        for i, (name, ckpt) in enumerate(available.items()):
            bar.progress(i / len(available), text=f"Running {name}…")
            try:
                model, _ = load_model(name, ckpt)
                t0 = time.perf_counter()
                prob, binary = infer(model, tensor, thr)
                results[name] = dict(prob=prob, binary=binary,
                                     ov=overlay(orig, binary),
                                     elapsed=time.perf_counter() - t0)
            except Exception as e:
                results[name] = {"error": str(e)}
        bar.progress(1.0, text="Done.")
        st.session_state["c_res"]  = results
        st.session_state["c_orig"] = orig

    if "c_res" not in st.session_state:
        footer(); return

    results = st.session_state["c_res"]
    orig    = st.session_state["c_orig"]
    ok_res  = {k: v for k, v in results.items() if "error" not in v}

    # ── Input ─────────────────────────────────────────────────────────────────
    sec("Input Image")
    ci, *_ = st.columns([1] + [1] * len(ok_res), gap="small")
    with ci: st.image(orig, use_container_width=True); img_cap("Input")

    # ── Masks ─────────────────────────────────────────────────────────────────
    sec("Predicted Masks")
    cols = st.columns(len(ok_res), gap="small")
    for col, (name, res) in zip(cols, ok_res.items()):
        with col:
            pct = 100.0 * res["binary"].sum() / res["binary"].size
            st.image(mask_pil(res["binary"]), use_container_width=True)
            img_cap(f"{name}  ·  {pct:.1f}%")

    # ── Overlays ──────────────────────────────────────────────────────────────
    sec("Overlays")
    cols2 = st.columns(len(ok_res), gap="small")
    for col, (name, res) in zip(cols2, ok_res.items()):
        with col:
            st.image(Image.fromarray(res["ov"]), use_container_width=True)
            img_cap(name)

    # ── Summary table ─────────────────────────────────────────────────────────
    sec("Inference Summary")
    rows = []
    for name, res in results.items():
        b = BENCHMARK.get(name, {})
        if "error" in res:
            rows.append({"Model": name, "Status": f"Error: {res['error']}",
                         "Time (ms)": "—", "Lesion %": "—", "Params": "—",
                         "Test Dice (ref)": "—", "Test IoU (ref)": "—"})
        else:
            pct = 100.0 * res["binary"].sum() / res["binary"].size
            rows.append({"Model": name, "Status": "OK",
                         "Time (ms)":       f"{res['elapsed']*1000:.0f}",
                         "Lesion %":        f"{pct:.1f}",
                         "Params":          b.get("Params", "—"),
                         "Test Dice (ref)": f"{b.get('Dice', 0):.4f}",
                         "Test IoU (ref)":  f"{b.get('IoU', 0):.4f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    footer()


# ═════════════════════════════════════════════════════════════════════════════
# Page — Metrics
# ═════════════════════════════════════════════════════════════════════════════

def page_metrics():
    st.title("Evaluation Metrics")
    st.markdown("Pre-computed results on the 390-image held-out test set.")
    st.divider()

    dark = st.get_option("theme.base") == "dark"

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Training Curves", "Visualizations", "Definitions"])

    # ── Tab 1 : Overview ──────────────────────────────────────────────────────
    with tab1:
        best = {k: max(BENCHMARK[m][k] for m in BENCHMARK)
                for k in ("Dice","IoU","Precision","Recall")}

        # Metric cards per model
        sec("Per-model results")
        cols = st.columns(3, gap="medium")
        for col, (name, b) in zip(cols, BENCHMARK.items()):
            with col:
                st.markdown(f"**{name}**")
                for key in ("Dice","IoU","Precision","Recall"):
                    is_best = abs(b[key]-best[key]) < 1e-4
                    col.markdown(stat_card(f"{b[key]:.4f}", key, best=is_best),
                                 unsafe_allow_html=True)
                    st.markdown("<div style='margin-top:5px'></div>",
                                unsafe_allow_html=True)

        # Full comparison dataframe
        sec("Full comparison table")
        df = pd.DataFrame(BENCHMARK).T
        num_cols = ["Dice","IoU","Precision","Recall"]
        st.dataframe(
            df.style.format({c: "{:.4f}" for c in num_cols})
                    .highlight_max(subset=num_cols, color="#bbf7d0"),
            use_container_width=True,
        )
        st.caption("Green = best value in each metric column.")

    # ── Tab 2 : Training curves ───────────────────────────────────────────────
    with tab2:
        histories = {n: h for n in ("SegNet","DeepLabV3")
                     if (h := load_history(n))}
        if histories:
            fig = build_curve_fig(histories, dark=dark)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig); buf.seek(0)
            st.image(buf, use_container_width=True)
            st.caption("Solid = validation · Dashed = training · "
                       "U-Net history not saved in epoch-by-epoch format.")
        else:
            # Fall back to static PNG files
            statics = [
                ("U-Net",     os.path.join(BASE_DIR,"outputs","training_curves.png")),
                ("SegNet",    os.path.join(BASE_DIR,"outputs","segnet","training_curves_segnet.png")),
                ("DeepLabV3", os.path.join(BASE_DIR,"outputs","deeplabv3","training_curves.png")),
            ]
            for name, path in statics:
                if os.path.isfile(path):
                    st.markdown(f"**{name}**")
                    st.image(path, use_container_width=True)

    # ── Tab 3 : Visualizations ────────────────────────────────────────────────
    with tab3:
        imgs = []
        pred_png = os.path.join(BASE_DIR,"outputs","predictions","test_predictions.png")
        if os.path.isfile(pred_png): imgs.append(("Test Predictions", pred_png))
        if os.path.isdir(CONFUSION_DIR):
            for f in sorted(os.listdir(CONFUSION_DIR)):
                if f.lower().endswith((".png",".jpg",".jpeg")):
                    label = os.path.splitext(f)[0].replace("_"," ").title()
                    imgs.append((label, os.path.join(CONFUSION_DIR, f)))

        if not imgs:
            st.info("No visualizations found in `outputs/` directory.")
        else:
            for i, (label, path) in enumerate(imgs):
                with st.expander(label, expanded=(i==0)):
                    st.image(path, use_container_width=True)
                    st.caption(f"`{os.path.relpath(path, BASE_DIR)}`")

    # ── Tab 4 : Definitions ───────────────────────────────────────────────────
    with tab4:
        st.markdown("### Metric Definitions")
        defs = [
            ("Dice (F1 Score)",
             "2|A∩B| / (|A|+|B|)",
             "Harmonic mean of precision and recall. Standard metric for medical "
             "segmentation. Range [0, 1] — higher is better."),
            ("IoU (Jaccard Index)",
             "|A∩B| / |A∪B|",
             "Ratio of intersection to union. More stringent than Dice — penalises "
             "false positives harder. Range [0, 1] — higher is better."),
            ("Precision",
             "TP / (TP + FP)",
             "Fraction of predicted lesion pixels that are truly lesion. "
             "High precision = few false positives."),
            ("Recall (Sensitivity)",
             "TP / (TP + FN)",
             "Fraction of true lesion pixels correctly detected. "
             "High recall = few missed lesion pixels."),
        ]
        for name, formula, desc in defs:
            with st.container(border=True):
                c1, c2 = st.columns([1, 2], gap="medium")
                with c1:
                    st.markdown(f"**{name}**")
                    st.code(formula, language=None)
                with c2:
                    st.markdown(desc)

    footer()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    page = sidebar()
    {
        "Home":       page_home,
        "Prediction": page_prediction,
        "Comparison": page_comparison,
        "Metrics":    page_metrics,
    }[page]()


if __name__ == "__main__":
    main()
