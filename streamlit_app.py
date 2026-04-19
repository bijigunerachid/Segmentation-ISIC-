"""
ISIC Skin Lesion Segmentation System
Streamlit Web Application
"""

import os
import sys
import json
import time
import warnings
import tempfile
from io import BytesIO

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st

warnings.filterwarnings("ignore")

# ─── Path Setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.model import UNet, SegNet, DeepLabV3
from src.utils import dice_score, iou_score

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEFAULT_CHECKPOINTS = {
    "U-Net":     os.path.join(BASE_DIR, "models", "best_unet_isic.pth"),
    "SegNet":    os.path.join(BASE_DIR, "models", "checkpoints", "best_model.pth"),
    "DeepLabV3": os.path.join(BASE_DIR, "models", "checkpoints", "best_model.pth"),
}

OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
METRICS_FILE    = os.path.join(OUTPUTS_DIR, "test_metrics.json")
CURVES_FILE     = os.path.join(OUTPUTS_DIR, "training_curves.png")
CONFUSION_DIR   = os.path.join(OUTPUTS_DIR, "confusion_matrix")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PARAMS = {
    "U-Net":     "31.0 M",
    "SegNet":    "29.4 M",
    "DeepLabV3": "39.6 M",
}

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISIC Skin Lesion Segmentation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', Arial, sans-serif;
}
.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #f8f9fa;
    border-right: 1px solid #e2e8f0;
}
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin: 1.5rem 0 0.6rem 0;
}
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 1.1rem 0.75rem;
    text-align: center;
}
.metric-card .val {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1e40af;
    line-height: 1.1;
}
.metric-card .lbl {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}
.img-label {
    text-align: center;
    font-size: 0.82rem;
    font-weight: 600;
    color: #4b5563;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.info-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.info-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #e5e7eb;
    vertical-align: top;
}
.info-table td:first-child {
    font-weight: 600;
    color: #374151;
    width: 42%;
    white-space: nowrap;
}
.disclaimer {
    font-size: 0.76rem;
    color: #9ca3af;
    border-top: 1px solid #e5e7eb;
    padding-top: 1rem;
    margin-top: 2.5rem;
    text-align: center;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ─── Utilities ────────────────────────────────────────────────────────────────

def section(title: str):
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)


def metric_card(value: str, label: str):
    return (
        f'<div class="metric-card">'
        f'<div class="val">{value}</div>'
        f'<div class="lbl">{label}</div>'
        f'</div>'
    )


def img_label(text: str):
    st.markdown(f'<p class="img-label">{text}</p>', unsafe_allow_html=True)


def display_footer():
    st.markdown(
        '<p class="disclaimer">This tool is for research and educational purposes only '
        'and is not intended for clinical diagnosis.</p>',
        unsafe_allow_html=True,
    )


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─── Image Processing ─────────────────────────────────────────────────────────

_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def preprocess(pil_img: Image.Image):
    """Returns (tensor [1,3,H,W], orig_rgb ndarray [H,W,3] uint8)."""
    img_rgb  = np.array(pil_img.convert("RGB"))
    orig_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor   = _transform(image=img_rgb)["image"].unsqueeze(0)
    return tensor, orig_rgb


def create_overlay(orig_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.45):
    """Blend a red lesion highlight over the original image."""
    orig_f   = orig_rgb.astype(np.float32)
    red      = np.zeros_like(orig_f)
    red[:, :, 0] = 220.0
    mask_3   = np.stack([binary_mask] * 3, axis=-1).astype(np.float32)
    blended  = orig_f * (1 - alpha * mask_3) + red * (alpha * mask_3)
    return blended.clip(0, 255).astype(np.uint8)


def binary_to_pil(binary_mask: np.ndarray) -> Image.Image:
    return Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")


def prob_to_pil(prob_map: np.ndarray) -> Image.Image:
    return Image.fromarray((prob_map * 255).clip(0, 255).astype(np.uint8), mode="L")


# ─── Model Loading ────────────────────────────────────────────────────────────

def _parse_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return ckpt, {}


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, checkpoint_path: str):
    """Load and cache a segmentation model. Returns (model, metadata_dict)."""
    if model_name == "U-Net":
        model = UNet(in_channels=3, out_channels=1)
    elif model_name == "SegNet":
        model = SegNet(in_channels=3, out_channels=1)
    elif model_name == "DeepLabV3":
        model = DeepLabV3(in_channels=3, out_channels=1, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict, meta = _parse_checkpoint(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, meta


def run_inference(model: torch.nn.Module, tensor: torch.Tensor, threshold: float = 0.5):
    """Returns (prob_map [H,W] float32, binary_mask [H,W] uint8)."""
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        prob   = torch.sigmoid(logits[0, 0]).cpu().numpy().astype(np.float32)
    binary = (prob >= threshold).astype(np.uint8)
    return prob, binary


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_segmentation_metrics(pred_binary: np.ndarray, gt_pil: Image.Image, threshold: float = 0.5):
    """Dice and IoU from a binary prediction and a GT mask PIL image."""
    gt_np  = np.array(gt_pil.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
    gt_bin = (gt_np > 127).astype(np.float32)
    p_bin  = pred_binary.astype(np.float32)

    intersection = (p_bin * gt_bin).sum()
    p_sum        = p_bin.sum()
    g_sum        = gt_bin.sum()
    smooth       = 1e-6

    dice = (2.0 * intersection + smooth) / (p_sum + g_sum + smooth)
    iou  = (intersection + smooth) / (p_sum + g_sum - intersection + smooth)
    return {"Dice": float(dice), "IoU": float(iou)}


# ─── Checkpoint Source Widget ─────────────────────────────────────────────────

def checkpoint_selector(model_name: str, key_prefix: str = ""):
    """Sidebar widget: returns a valid checkpoint path or None."""
    source = st.radio(
        "Checkpoint source",
        ["Default path", "Upload file"],
        index=0,
        key=f"{key_prefix}_src",
    )

    if source == "Default path":
        default = DEFAULT_CHECKPOINTS[model_name]
        rel     = os.path.relpath(default, BASE_DIR)
        exists  = os.path.isfile(default)
        st.caption(f"`{rel}`")
        if exists:
            st.success("Checkpoint found")
        else:
            st.error("File not found — train the model or upload a checkpoint")
        return default if exists else None

    uploaded = st.file_uploader(
        "Upload .pth file",
        type=["pth", "pt"],
        key=f"{key_prefix}_upload",
    )
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tmp.write(uploaded.read())
        tmp.flush()
        return tmp.name
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Page: Home
# ═════════════════════════════════════════════════════════════════════════════

def page_home():
    st.title("ISIC Skin Lesion Segmentation System")
    st.markdown(
        "Automatic pixel-wise segmentation of dermoscopic skin lesion images using "
        "deep learning models trained on the **ISIC 2018 Task 1** benchmark dataset."
    )
    st.divider()

    left, right = st.columns([3, 2], gap="large")

    with left:
        section("Overview")
        st.markdown("""
This system performs binary semantic segmentation of dermoscopic images, classifying
each pixel as either lesion tissue or surrounding healthy skin. It serves as a
research interface for evaluating and comparing deep learning segmentation approaches
in a medical imaging context.

Models are trained end-to-end using a combined **Dice-BCE loss**. Performance is
quantified with the **Dice coefficient** and **Intersection over Union (IoU)**,
the standard metrics for medical image segmentation benchmarks.
        """)

        section("Dataset")
        device_str = "GPU — " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.markdown(f"""
<table class="info-table">
<tr><td>Dataset</td><td>ISIC 2018 — Skin Lesion Analysis Toward Melanoma Detection</td></tr>
<tr><td>Task</td><td>Task 1 — Lesion Boundary Segmentation</td></tr>
<tr><td>Modality</td><td>Dermoscopy (epiluminescence microscopy)</td></tr>
<tr><td>Input format</td><td>RGB dermoscopic images, JPEG</td></tr>
<tr><td>Output format</td><td>Binary segmentation mask (lesion / background)</td></tr>
<tr><td>Input resolution</td><td>Resized to 256 × 256 pixels</td></tr>
<tr><td>Normalization</td><td>ImageNet mean and standard deviation</td></tr>
<tr><td>Loss function</td><td>Dice Loss + Binary Cross-Entropy</td></tr>
</table>
        """, unsafe_allow_html=True)

        section("System")
        st.markdown(f"""
<table class="info-table">
<tr><td>Compute device</td><td>{device_str}</td></tr>
<tr><td>Framework</td><td>PyTorch {torch.__version__}</td></tr>
<tr><td>Input resolution</td><td>256 × 256 px</td></tr>
</table>
        """, unsafe_allow_html=True)

    with right:
        section("Segmentation Models")

        descriptions = {
            "U-Net": (
                "Encoder-decoder architecture with symmetric skip connections. "
                "The reference standard for medical image segmentation since 2015.",
                "Features: 64 → 128 → 256 → 512 — Params: 31.0 M"
            ),
            "SegNet": (
                "VGG-style encoder paired with an index-based max-unpooling decoder. "
                "Retains precise spatial boundary information through pooling indices.",
                "5-stage encoder/decoder — Params: 29.4 M"
            ),
            "DeepLabV3": (
                "ResNet-50 backbone with Atrous Spatial Pyramid Pooling (ASPP) for "
                "multi-scale contextual feature extraction via dilated convolutions.",
                "Pretrained backbone, ASPP module — Params: 39.6 M"
            ),
        }

        for name, (desc, note) in descriptions.items():
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(desc)
                st.markdown(
                    f"<small style='color:#6b7280;font-size:0.8rem'>{note}</small>",
                    unsafe_allow_html=True,
                )

        section("Navigation")
        st.markdown("""
<table class="info-table">
<tr><td>Prediction</td><td>Run inference on a single image with a chosen model</td></tr>
<tr><td>Comparison</td><td>Compare all three models side by side on one image</td></tr>
<tr><td>Metrics</td><td>View evaluation results and training diagnostics</td></tr>
</table>
        """, unsafe_allow_html=True)

    display_footer()


# ═════════════════════════════════════════════════════════════════════════════
# Page: Prediction
# ═════════════════════════════════════════════════════════════════════════════

def page_prediction():
    st.title("Prediction")
    st.markdown(
        "Upload a dermoscopic image, select a model and checkpoint, "
        "then run segmentation inference."
    )
    st.divider()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Model")
        model_name = st.selectbox("Architecture", ["U-Net", "SegNet", "DeepLabV3"], key="pred_model")
        checkpoint_path = checkpoint_selector(model_name, key_prefix="pred")

        st.markdown("---")
        st.markdown("### Inference Settings")
        threshold = st.slider("Segmentation threshold", 0.10, 0.90, 0.50, 0.05, key="pred_thr")

        st.markdown("---")
        st.markdown("### Ground Truth (optional)")
        st.caption("Upload a binary mask to compute Dice and IoU metrics.")
        gt_file = st.file_uploader("Mask image", type=["png", "jpg", "jpeg"], key="pred_gt")

    # ── Image upload ──────────────────────────────────────────────────────────
    uploaded_img = st.file_uploader(
        "Input dermoscopic image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="pred_img",
    )

    if not uploaded_img:
        st.info("Upload an input image to begin.")
        display_footer()
        return

    pil_img = Image.open(uploaded_img).convert("RGB")

    if not checkpoint_path:
        st.warning("Provide a valid model checkpoint to proceed.")
        display_footer()
        return

    run_btn = st.button("Run Inference", type="primary")

    if run_btn:
        with st.spinner(f"Loading {model_name} and running inference..."):
            try:
                model, meta = load_model(model_name, checkpoint_path)
                tensor, orig_rgb = preprocess(pil_img)
                t0 = time.perf_counter()
                prob_map, binary_mask = run_inference(model, tensor, threshold)
                elapsed = time.perf_counter() - t0
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                display_footer()
                return

        st.session_state["pred_result"] = {
            "orig_rgb":    orig_rgb,
            "prob_map":    prob_map,
            "binary_mask": binary_mask,
            "elapsed":     elapsed,
            "meta":        meta,
            "model_name":  model_name,
        }

    if "pred_result" not in st.session_state:
        display_footer()
        return

    res         = st.session_state["pred_result"]
    orig_rgb    = res["orig_rgb"]
    prob_map    = res["prob_map"]
    binary_mask = res["binary_mask"]
    elapsed     = res["elapsed"]
    meta        = res["meta"]
    overlay     = create_overlay(orig_rgb, binary_mask)

    # ── Results display ───────────────────────────────────────────────────────
    section("Segmentation Results")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.image(orig_rgb, use_container_width=True)
        img_label("Input Image")
    with c2:
        st.image(binary_to_pil(binary_mask), use_container_width=True)
        img_label("Predicted Mask")
    with c3:
        st.image(Image.fromarray(overlay), use_container_width=True)
        img_label("Overlay")

    # ── Statistics ────────────────────────────────────────────────────────────
    section("Prediction Statistics")
    lesion_px  = int(binary_mask.sum())
    total_px   = binary_mask.size
    lesion_pct = 100.0 * lesion_px / total_px
    max_conf   = float(prob_map.max())

    s1, s2, s3, s4 = st.columns(4, gap="medium")
    for col, val, lbl in [
        (s1, f"{elapsed * 1000:.1f} ms", "Inference time"),
        (s2, f"{lesion_pct:.1f}%",       "Lesion area"),
        (s3, f"{lesion_px:,}",            "Lesion pixels"),
        (s4, f"{max_conf:.3f}",           "Max confidence"),
    ]:
        with col:
            st.markdown(metric_card(val, lbl), unsafe_allow_html=True)

    # ── Ground truth metrics ──────────────────────────────────────────────────
    if gt_file:
        gt_pil  = Image.open(gt_file)
        metrics = compute_segmentation_metrics(binary_mask, gt_pil, threshold)
        section("Evaluation Metrics")
        m1, m2, _, _ = st.columns(4, gap="medium")
        with m1:
            st.markdown(metric_card(f"{metrics['Dice']:.4f}", "Dice Coefficient"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card(f"{metrics['IoU']:.4f}", "IoU Score"), unsafe_allow_html=True)

    # ── Checkpoint metadata ───────────────────────────────────────────────────
    if meta:
        with st.expander("Checkpoint information"):
            rows = []
            if "epoch" in meta:
                rows.append(f"<tr><td>Epoch</td><td>{meta['epoch']}</td></tr>")
            if "val_dice" in meta:
                rows.append(f"<tr><td>Val Dice (saved)</td><td>{meta['val_dice']:.4f}</td></tr>")
            if "val_iou" in meta:
                rows.append(f"<tr><td>Val IoU (saved)</td><td>{meta['val_iou']:.4f}</td></tr>")
            if "config" in meta and isinstance(meta["config"], dict):
                cfg = meta["config"]
                if "img_size" in cfg:
                    rows.append(f"<tr><td>Training image size</td><td>{cfg['img_size']}</td></tr>")
                if "learning_rate" in cfg:
                    rows.append(f"<tr><td>Learning rate</td><td>{cfg['learning_rate']}</td></tr>")
            if rows:
                st.markdown(
                    f'<table class="info-table">{"".join(rows)}</table>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No structured metadata available in this checkpoint.")

    # ── Export ────────────────────────────────────────────────────────────────
    section("Export")
    d1, d2 = st.columns(2, gap="medium")
    with d1:
        st.download_button(
            "Download Predicted Mask",
            data=pil_to_bytes(binary_to_pil(binary_mask)),
            file_name="predicted_mask.png",
            mime="image/png",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download Overlay",
            data=pil_to_bytes(Image.fromarray(overlay)),
            file_name="overlay.png",
            mime="image/png",
            use_container_width=True,
        )

    display_footer()


# ═════════════════════════════════════════════════════════════════════════════
# Page: Comparison
# ═════════════════════════════════════════════════════════════════════════════

def page_comparison():
    st.title("Model Comparison")
    st.markdown(
        "Upload a single dermoscopic image to run all three models simultaneously "
        "and compare their segmentation outputs side by side."
    )
    st.divider()

    with st.sidebar:
        st.markdown("### Checkpoint Paths")
        ckpt_paths = {}
        for name in ["U-Net", "SegNet", "DeepLabV3"]:
            with st.expander(name, expanded=(name == "U-Net")):
                default = DEFAULT_CHECKPOINTS[name]
                exists  = os.path.isfile(default)
                rel     = os.path.relpath(default, BASE_DIR)
                st.caption(f"`{rel}`")
                if exists:
                    st.success("Found")
                    ckpt_paths[name] = default
                else:
                    st.warning("Not found")
                    up = st.file_uploader(f"Upload {name} .pth", type=["pth", "pt"], key=f"cmp_{name}")
                    if up:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
                        tmp.write(up.read())
                        tmp.flush()
                        ckpt_paths[name] = tmp.name

        st.markdown("---")
        threshold = st.slider("Segmentation threshold", 0.10, 0.90, 0.50, 0.05, key="cmp_thr")

    uploaded_img = st.file_uploader(
        "Input dermoscopic image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="cmp_img",
    )

    if not uploaded_img:
        st.info("Upload an image to begin.")
        display_footer()
        return

    available = {k: v for k, v in ckpt_paths.items() if os.path.isfile(v)}
    if not available:
        st.warning("No valid checkpoints found. Provide at least one checkpoint in the sidebar.")
        display_footer()
        return

    pil_img = Image.open(uploaded_img).convert("RGB")
    run_btn = st.button("Run All Models", type="primary")

    if run_btn:
        tensor, orig_rgb = preprocess(pil_img)
        results = {}

        progress = st.progress(0, text="Running inference...")
        n_models = len(available)

        for i, (name, ckpt) in enumerate(available.items()):
            progress.progress((i) / n_models, text=f"Running {name}...")
            try:
                model, _ = load_model(name, ckpt)
                t0       = time.perf_counter()
                prob, binary = run_inference(model, tensor, threshold)
                elapsed  = time.perf_counter() - t0
                results[name] = {
                    "prob":    prob,
                    "binary":  binary,
                    "overlay": create_overlay(orig_rgb, binary),
                    "elapsed": elapsed,
                }
            except Exception as exc:
                results[name] = {"error": str(exc)}

        progress.progress(1.0, text="Done.")
        st.session_state["cmp_results"]  = results
        st.session_state["cmp_orig_rgb"] = orig_rgb

    if "cmp_results" not in st.session_state:
        display_footer()
        return

    results  = st.session_state["cmp_results"]
    orig_rgb = st.session_state.get("cmp_orig_rgb")

    # ── Input image ───────────────────────────────────────────────────────────
    section("Input Image")
    col_img, _ = st.columns([1, 3])
    with col_img:
        st.image(orig_rgb, use_container_width=True)
        img_label("Input")

    # ── Predicted masks ───────────────────────────────────────────────────────
    section("Predicted Masks")
    cols = st.columns(len(results), gap="medium")
    for col, (name, res) in zip(cols, results.items()):
        with col:
            if "error" in res:
                st.error(f"{name}: {res['error']}")
            else:
                st.image(binary_to_pil(res["binary"]), use_container_width=True)
                img_label(name)
                pct = 100.0 * res["binary"].sum() / res["binary"].size
                st.caption(f"Lesion: {pct:.1f}%  |  {res['elapsed']*1000:.0f} ms")

    # ── Overlays ──────────────────────────────────────────────────────────────
    section("Overlay Comparison")
    cols2 = st.columns(len(results), gap="medium")
    for col, (name, res) in zip(cols2, results.items()):
        with col:
            if "error" not in res:
                st.image(Image.fromarray(res["overlay"]), use_container_width=True)
                img_label(f"{name} — Overlay")

    # ── Inference time table ──────────────────────────────────────────────────
    section("Inference Summary")
    rows = []
    for name, res in results.items():
        if "error" not in res:
            pct     = 100.0 * res["binary"].sum() / res["binary"].size
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td>{res['elapsed']*1000:.1f} ms</td>"
                f"<td>{pct:.1f}%</td>"
                f"<td>{MODEL_PARAMS.get(name, '—')}</td></tr>"
            )
    if rows:
        st.markdown(
            '<table class="info-table">'
            "<tr><td><b>Model</b></td><td><b>Inference time</b></td>"
            "<td><b>Lesion area</b></td><td><b>Parameters</b></td></tr>"
            + "".join(rows)
            + "</table>",
            unsafe_allow_html=True,
        )

    display_footer()


# ═════════════════════════════════════════════════════════════════════════════
# Page: Metrics
# ═════════════════════════════════════════════════════════════════════════════

def page_metrics():
    st.title("Evaluation Metrics")
    st.markdown(
        "Quantitative evaluation results, training diagnostics, and segmentation "
        "quality visualizations generated during model training and testing."
    )
    st.divider()

    # ── Test metrics ──────────────────────────────────────────────────────────
    section("Test Set Results")

    if os.path.isfile(METRICS_FILE):
        with open(METRICS_FILE) as f:
            metrics = json.load(f)

        def _fmt(v):
            try:
                return f"{float(v):.4f}"
            except (TypeError, ValueError):
                return str(v)

        # Display each model's metrics
        if isinstance(metrics, dict):
            first_val = next(iter(metrics.values()), None)

            if isinstance(first_val, dict):
                # Format: {"U-Net": {"dice": 0.87, "iou": 0.79}, ...}
                model_names = list(metrics.keys())
                cols = st.columns(len(model_names), gap="medium")
                for col, mname in zip(cols, model_names):
                    with col:
                        st.markdown(f"**{mname}**")
                        m = metrics[mname]
                        dice_val = m.get("dice", m.get("Dice", m.get("dice_score", "—")))
                        iou_val  = m.get("iou",  m.get("IoU",  m.get("iou_score",  "—")))
                        st.markdown(metric_card(_fmt(dice_val), "Dice"), unsafe_allow_html=True)
                        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
                        st.markdown(metric_card(_fmt(iou_val),  "IoU"),  unsafe_allow_html=True)
            else:
                # Format: {"dice": 0.87, "iou": 0.79, ...} — single model
                dice_val = metrics.get("dice", metrics.get("Dice", metrics.get("dice_score", "—")))
                iou_val  = metrics.get("iou",  metrics.get("IoU",  metrics.get("iou_score",  "—")))
                m1, m2, _, _ = st.columns(4, gap="medium")
                with m1:
                    st.markdown(metric_card(_fmt(dice_val), "Dice Coefficient"), unsafe_allow_html=True)
                with m2:
                    st.markdown(metric_card(_fmt(iou_val), "IoU Score"), unsafe_allow_html=True)

                with st.expander("Full metrics JSON"):
                    st.json(metrics)
    else:
        st.info(
            f"No test metrics file found at `{os.path.relpath(METRICS_FILE, BASE_DIR)}`."
            " Run the evaluation script to generate it."
        )

    # ── Training curves ───────────────────────────────────────────────────────
    section("Training Curves")

    if os.path.isfile(CURVES_FILE):
        st.image(CURVES_FILE, use_container_width=True)
        st.caption(f"Source: `{os.path.relpath(CURVES_FILE, BASE_DIR)}`")
    else:
        st.info(
            f"No training curves image found at `{os.path.relpath(CURVES_FILE, BASE_DIR)}`."
        )

    # ── Confusion / segmentation visualizations ───────────────────────────────
    section("Segmentation Visualizations")

    conf_images = []
    if os.path.isdir(CONFUSION_DIR):
        for fname in sorted(os.listdir(CONFUSION_DIR)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                conf_images.append(os.path.join(CONFUSION_DIR, fname))

    pred_img = os.path.join(OUTPUTS_DIR, "predictions", "test_predictions.png")
    if os.path.isfile(pred_img):
        conf_images.insert(0, pred_img)

    if conf_images:
        for img_path in conf_images:
            rel  = os.path.relpath(img_path, BASE_DIR)
            name = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ").title()
            with st.expander(name, expanded=(img_path == conf_images[0])):
                st.image(img_path, use_container_width=True)
                st.caption(f"`{rel}`")
    else:
        st.info(
            "No segmentation visualizations found. "
            "Run a prediction or the evaluation notebook to generate them."
        )

    # ── Metric definitions ────────────────────────────────────────────────────
    section("Metric Definitions")
    st.markdown("""
<table class="info-table">
<tr>
  <td>Dice coefficient</td>
  <td>
    Measures overlap between prediction and ground truth.
    <i>Dice = 2|A ∩ B| / (|A| + |B|)</i>.
    Range [0, 1], higher is better. Equivalent to F1 score.
  </td>
</tr>
<tr>
  <td>IoU (Jaccard Index)</td>
  <td>
    Ratio of intersection to union of predicted and true masks.
    <i>IoU = |A ∩ B| / |A ∪ B|</i>.
    Range [0, 1], higher is better. More stringent than Dice.
  </td>
</tr>
</table>
    """, unsafe_allow_html=True)

    display_footer()


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ═════════════════════════════════════════════════════════════════════════════

def sidebar_nav():
    with st.sidebar:
        st.markdown("## ISIC Segmentation")
        st.markdown(
            "<small style='color:#6b7280'>ISIC 2018 · Task 1 · Deep Learning</small>",
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.radio(
            "Navigation",
            ["Home", "Prediction", "Comparison", "Metrics"],
            label_visibility="collapsed",
        )

        st.divider()
        device_label = "GPU" if torch.cuda.is_available() else "CPU"
        st.markdown(
            f"<small style='color:#9ca3af'>Device: {device_label} &nbsp;|&nbsp; "
            f"PyTorch {torch.__version__}</small>",
            unsafe_allow_html=True,
        )

    return page


# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    page = sidebar_nav()

    if page == "Home":
        page_home()
    elif page == "Prediction":
        page_prediction()
    elif page == "Comparison":
        page_comparison()
    elif page == "Metrics":
        page_metrics()


if __name__ == "__main__":
    main()
