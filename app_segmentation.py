import os
import sys
import threading
import datetime
import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Ajoute la racine du projet au path ──────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Constantes ──
TAB_RESULTS = "Résultats"
TAB_HISTORY = "Historique"

try:
    from src.model  import UNet
    from src.utils  import dice_score, iou_score
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    # Import des architectures depuis le notebook
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models.segmentation as seg_models
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# ── Utiliser UNetApp au lieu de UNet de src.model ──
class UNet(nn.Module):
    """U-Net pour l'application - même architecture que les notebooks."""
    def __init__(self, n_channels=3, n_classes=1, features=[64,128,256,512]):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        in_channels = n_channels
        for f in features:
            self.downs.append(DoubleConv(in_channels, f)); in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.final = nn.Conv2d(features[0], n_classes, 1)
    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); skip = skips[i//2]
            if x.shape != skip.shape: x = F.interpolate(x, size=skip.shape[2:])
            x = self.ups[i+1](torch.cat([skip, x], dim=1))
        return self.final(x)

# ── Thème ────────────────────────────────────────────────────
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# ── Constantes ───────────────────────────────────────────────
MODEL_PATHS = {
    'U-Net': os.path.join(ROOT, "models", "best_unet_isic.pth"),
    'SegNet': os.path.join(ROOT, "models", "checkpoints", "best_model.pth"),  # SegNet si entraîné
    'DeepLabV3': os.path.join(ROOT, "models", "checkpoints", "best_model.pth")  # DeepLabV3 si entraîné
}
IMG_SIZE     = 256  # Changé à 256 pour correspondre aux notebooks
HISTORY_FILE = os.path.join(ROOT, "outputs", "history.json")
PRED_DIR     = os.path.join(ROOT, "outputs", "predictions")

# ════════════════════════════════════════════════════════════
# Définitions des architectures (depuis le notebook)
# ════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class UNetApp(nn.Module):
    """U-Net pour l'application."""
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        for f in features:
            self.downs.append(DoubleConv(in_channels, f)); in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.final = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); skip = skips[i//2]
            if x.shape != skip.shape: x = F.interpolate(x, size=skip.shape[2:])
            x = self.ups[i+1](torch.cat([skip, x], dim=1))
        return self.final(x)

class SegNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n=2):
        super().__init__()
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(in_ch if i==0 else out_ch, out_ch, 3, padding=1),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class SegNetApp(nn.Module):
    """SegNet pour l'application."""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = SegNetBlock(in_channels,64,2); self.enc2 = SegNetBlock(64,128,2)
        self.enc3 = SegNetBlock(128,256,3);         self.enc4 = SegNetBlock(256,512,3)
        self.enc5 = SegNetBlock(512,512,3)
        self.dec5 = SegNetBlock(512,512,3); self.dec4 = SegNetBlock(512,256,3)
        self.dec3 = SegNetBlock(256,128,3); self.dec2 = SegNetBlock(128,64,2)
        self.dec1 = SegNetBlock(64,64,2)
        self.pool   = nn.MaxPool2d(2,2,return_indices=True)
        self.unpool = nn.MaxUnpool2d(2,2)
        self.final  = nn.Conv2d(64, out_channels, 1)
    def forward(self, x):
        x1=self.enc1(x); x,i1=self.pool(x1)
        x2=self.enc2(x); x,i2=self.pool(x2)
        x3=self.enc3(x); x,i3=self.pool(x3)
        x4=self.enc4(x); x,i4=self.pool(x4)
        x5=self.enc5(x); x,i5=self.pool(x5)
        x=self.unpool(x,i5,output_size=x5.size()); x=self.dec5(x)
        x=self.unpool(x,i4,output_size=x4.size()); x=self.dec4(x)
        x=self.unpool(x,i3,output_size=x3.size()); x=self.dec3(x)
        x=self.unpool(x,i2,output_size=x2.size()); x=self.dec2(x)
        x=self.unpool(x,i1,output_size=x1.size()); x=self.dec1(x)
        return self.final(x)

class DeepLabV3Wrapper(nn.Module):
    """DeepLabV3 pour l'application."""
    def __init__(self, out_channels=1, pretrained=True):
        super().__init__()
        weights = seg_models.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = seg_models.deeplabv3_resnet50(weights=weights, aux_loss=True)
        self.model.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)
    def forward(self, x):
        out = self.model(x)['out']
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

# ── Prétraitement ────────────────────────────────────────────
def get_transform(img_size=128):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    """Tensor [3,H,W] → numpy RGB [H,W,3] uint8."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * STD + MEAN).clip(0, 1)
    return (img * 255).astype(np.uint8)

# ════════════════════════════════════════════════════════════
# Application principale
# ════════════════════════════════════════════════════════════
class SegApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Segmentation des Lésions Cutanées — U-Net ISIC 2018")
        self.geometry("1350x800")
        self.minsize(1100, 700)

        # ── État ──
        self.model        = None
        self.model_type   = ctk.StringVar(value="U-Net")  # Nouveau: type de modèle
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path    = ctk.StringVar(value=MODEL_PATHS["U-Net"])  # Chemin par défaut pour U-Net
        self.threshold    = ctk.DoubleVar(value=0.5)
        self.img_size_var = ctk.IntVar(value=IMG_SIZE)

        self.orig_rgb    = None   # numpy RGB original
        self.tensor_img  = None   # tensor normalisé
        self.pred_mask   = None   # numpy [H,W] float 0-1
        self.gt_mask     = None   # numpy [H,W] binaire (optionnel)
        self.current_img_path = ""
        self.current_gt_path  = ""

        self.history     = []     # liste de dicts
        self._load_history()

        # Lier le changement de type de modèle à la mise à jour du chemin
        self.model_type.trace_add("write", self._on_model_type_change)

        os.makedirs(PRED_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

        self._build_ui()
        self._check_imports()

    # ═══════════════════════════════════════════════════════
    # Interface utilisateur
    # ═══════════════════════════════════════════════════════
    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ── Sidebar gauche ──
        self._build_sidebar()

        # ── Zone principale (droite) ──
        self._build_main()

    # ── Sidebar ─────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkScrollableFrame(self, width=300, corner_radius=0,
                                     label_text="")
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_columnconfigure(0, weight=1)
        self.sidebar = sb

        # Titre
        ctk.CTkLabel(sb, text="Segmentation\nLésions Cutanées",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     justify="left").grid(row=0, column=0, padx=20, pady=(20, 4), sticky="w")
        ctk.CTkLabel(sb, text="U-Net · ISIC 2018 · PyTorch",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        self._sep(sb, 2)

        # ── Section : Modèle ──
        self._section(sb, 3, "Modèle")

        # Type de modèle
        ctk.CTkLabel(sb, text="Architecture :",
                     font=ctk.CTkFont(size=12)).grid(
            row=4, column=0, padx=20, pady=(0, 4), sticky="w")

        model_combo = ctk.CTkComboBox(
            sb, values=["U-Net", "SegNet", "DeepLabV3"],
            variable=self.model_type, command=self._on_model_type_change,
            font=ctk.CTkFont(size=12), height=32)
        model_combo.grid(row=5, column=0, padx=20, pady=(0, 8), sticky="ew")

        # Statut modèle
        self.model_status = ctk.CTkLabel(
            sb, text="Non chargé", font=ctk.CTkFont(size=12),
            text_color="gray", wraplength=260, justify="left")
        self.model_status.grid(row=6, column=0, padx=20, pady=(0, 8), sticky="w")

        # Chemin checkpoint
        ctk.CTkLabel(sb, text="Checkpoint (.pth) :",
                     font=ctk.CTkFont(size=12)).grid(
            row=7, column=0, padx=20, pady=(0, 4), sticky="w")

        ckpt_frame = ctk.CTkFrame(sb, fg_color="transparent")
        ckpt_frame.grid(row=8, column=0, padx=20, pady=(0, 8), sticky="ew")
        ckpt_frame.grid_columnconfigure(0, weight=1)

        self.ckpt_entry = ctk.CTkEntry(
            ckpt_frame, textvariable=self.ckpt_path,
            font=ctk.CTkFont(size=10), height=30)
        self.ckpt_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(ckpt_frame, text="...", width=32, height=30,
                      command=self._browse_ckpt).grid(row=0, column=1)

        ctk.CTkButton(sb, text="Charger le modèle", height=36,
                      command=self._load_model_thread).grid(
            row=9, column=0, padx=20, pady=(0, 4), sticky="ew")

        # Device info
        self.device_label = ctk.CTkLabel(
            sb, text=f"Device : {self.device}",
            font=ctk.CTkFont(size=11), text_color="gray")
        self.device_label.grid(row=10, column=0, padx=20, pady=(0, 12), sticky="w")

        self._sep(sb, 11)

        # ── Section : Image ──
        self._section(sb, 12, "Image d'entrée")

        ctk.CTkButton(sb, text="Ouvrir une image (.jpg / .png)",
                      height=36, command=self._open_image).grid(
            row=13, column=0, padx=20, pady=(0, 6), sticky="ew")

        ctk.CTkButton(sb, text="Charger masque GT (optionnel)",
                      height=34, fg_color="transparent",
                      border_width=1, command=self._open_gt).grid(
            row=14, column=0, padx=20, pady=(0, 4), sticky="ew")

        self.img_info = ctk.CTkLabel(
            sb, text="Aucune image chargée",
            font=ctk.CTkFont(size=11), text_color="gray",
            wraplength=260, justify="left")
        self.img_info.grid(row=15, column=0, padx=20, pady=(0, 12), sticky="w")

        self._sep(sb, 16)

        # ── Section : Paramètres ──
        self._section(sb, 17, "Paramètres")

        # Seuil
        ctk.CTkLabel(sb, text="Seuil de segmentation :",
                     font=ctk.CTkFont(size=12)).grid(
            row=18, column=0, padx=20, pady=(0, 2), sticky="w")

        thr_frame = ctk.CTkFrame(sb, fg_color="transparent")
        thr_frame.grid(row=19, column=0, padx=20, pady=(0, 2), sticky="ew")
        thr_frame.grid_columnconfigure(0, weight=1)

        self.thr_slider = ctk.CTkSlider(
            thr_frame, from_=0.1, to=0.9, number_of_steps=80,
            variable=self.threshold, command=self._on_threshold_change)
        self.thr_slider.grid(row=0, column=0, sticky="ew")

        self.thr_label = ctk.CTkLabel(
            sb, text="0.50",
            font=ctk.CTkFont(size=13, weight="bold"))
        self.thr_label.grid(row=20, column=0, padx=20, pady=(0, 10), sticky="w")

        # Taille image
        ctk.CTkLabel(sb, text="Taille d'entrée :",
                     font=ctk.CTkFont(size=12)).grid(
            row=21, column=0, padx=20, pady=(0, 4), sticky="w")

        size_frame = ctk.CTkFrame(sb, fg_color="transparent")
        size_frame.grid(row=22, column=0, padx=20, pady=(0, 10), sticky="w")
        for size in [128, 256]:
            ctk.CTkRadioButton(
                size_frame, text=f"{size}×{size}",
                variable=self.img_size_var, value=size,
                font=ctk.CTkFont(size=12)).pack(side="left", padx=8)

        self._sep(sb, 23)

        # ── Section : Actions ──
        self._section(sb, 24, "Actions")

        self.predict_btn = ctk.CTkButton(
            sb, text="Segmenter", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled", command=self._predict_thread)
        self.predict_btn.grid(row=25, column=0, padx=20, pady=(0, 6), sticky="ew")

        ctk.CTkButton(sb, text="Sauvegarder le masque prédit",
                      height=34, fg_color="transparent", border_width=1,
                      command=self._save_mask).grid(
            row=26, column=0, padx=20, pady=(0, 4), sticky="ew")

        ctk.CTkButton(sb, text="Sauvegarder l'overlay",
                      height=34, fg_color="transparent", border_width=1,
                      command=self._save_overlay).grid(
            row=27, column=0, padx=20, pady=(0, 12), sticky="ew")

        self._sep(sb, 28)

        # ── Section : Métriques ──
        self._section(sb, 29, "Métriques")

        metrics_frame = ctk.CTkFrame(sb, corner_radius=8)
        metrics_frame.grid(row=30, column=0, padx=20, pady=(0, 12), sticky="ew")
        metrics_frame.grid_columnconfigure((0, 1), weight=1)

        for j, (lbl, attr) in enumerate([("Dice Score", "dice_val"), ("IoU Score", "iou_val")]):
            ctk.CTkLabel(metrics_frame, text=lbl,
                         font=ctk.CTkFont(size=11), text_color="gray").grid(
                row=0, column=j, padx=12, pady=(10, 2))
            lv = ctk.CTkLabel(metrics_frame, text="—",
                              font=ctk.CTkFont(size=20, weight="bold"))
            lv.grid(row=1, column=j, padx=12, pady=(0, 10))
            setattr(self, attr, lv)

        self.metrics_note = ctk.CTkLabel(
            sb, text="Charger un masque GT pour calculer les métriques",
            font=ctk.CTkFont(size=10), text_color="gray",
            wraplength=260, justify="left")
        self.metrics_note.grid(row=31, column=0, padx=20, pady=(0, 20), sticky="w")

        # Bouton thème
        ctk.CTkButton(sb, text="Thème sombre / clair",
                      height=30, fg_color="transparent", border_width=1,
                      command=self._toggle_theme).grid(
            row=32, column=0, padx=20, pady=(0, 16), sticky="ew")

    # ── Zone principale ──────────────────────────────────────
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        main.grid_rowconfigure(1, weight=1)
        main.grid_rowconfigure(2, weight=0)
        main.grid_columnconfigure(0, weight=1)

        # ── Barre de statut haut ──
        self.status_bar = ctk.CTkLabel(
            main, text="Bienvenue — Chargez le modèle puis une image pour commencer.",
            font=ctk.CTkFont(size=12), text_color="gray",
            anchor="w")
        self.status_bar.grid(row=0, column=0, padx=16, pady=(12, 4), sticky="ew")

        # ── Notebook (onglets) ──
        self.tabs = ctk.CTkTabview(main, corner_radius=10)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))

        for tab in [TAB_RESULTS, TAB_HISTORY]:
            self.tabs.add(tab)

        self._build_results_tab()
        self._build_history_tab()

        # ── Barre de progression ──
        self.progress = ctk.CTkProgressBar(main, height=6)
        self.progress.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.progress.set(0)

    def _build_results_tab(self):
        tab = self.tabs.tab(TAB_RESULTS)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure((0, 1, 2), weight=1)

        # Titres colonnes
        for j, title in enumerate(["Image originale", "Masque prédit", "Overlay (lésion)"]):
            ctk.CTkLabel(tab, text=title,
                         font=ctk.CTkFont(size=13, weight="bold")).grid(
                row=0, column=j, padx=8, pady=(8, 4))

        # Canvases d'affichage
        self.canvases = []
        self.canvas_imgs = [None, None, None]

        for j in range(3):
            frame = ctk.CTkFrame(tab, corner_radius=10,
                                  border_width=1,
                                  border_color=("gray80", "gray30"))
            frame.grid(row=1, column=j, padx=8, pady=(0, 8), sticky="nsew")
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)

            canvas = tk.Canvas(frame, bg="#1a1a2e" if ctk.get_appearance_mode() == "Dark" else "#f0f0f0",
                               highlightthickness=0)
            canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            canvas.bind("<Configure>", lambda e, idx=j: self._redraw_canvas(idx))
            self.canvases.append(canvas)

        # Placeholder texte sur les canvases
        for c in self.canvases:
            c.create_text(5, 5, anchor="nw",
                          text="Aucune image",
                          fill="gray", font=("Arial", 11), tags="placeholder")

    def _build_history_tab(self):
        tab = self.tabs.tab(TAB_HISTORY)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Frame scrollable pour l'historique
        self.hist_frame = ctk.CTkScrollableFrame(tab, label_text="")
        self.hist_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.hist_frame.grid_columnconfigure(0, weight=1)

        # Header
        hdr = ctk.CTkFrame(self.hist_frame, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        hdr.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(hdr, text="Historique des analyses",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=12, pady=8, sticky="w")
        ctk.CTkButton(hdr, text="Effacer", width=80, height=28,
                      fg_color="transparent", border_width=1,
                      command=self._clear_history).grid(
            row=0, column=1, padx=12, pady=8, sticky="e")

        self.hist_rows_frame = ctk.CTkFrame(self.hist_frame, fg_color="transparent")
        self.hist_rows_frame.grid(row=1, column=0, sticky="ew")
        self.hist_rows_frame.grid_columnconfigure(0, weight=1)

        self._refresh_history_ui()

    # ── Helpers UI ───────────────────────────────────────────
    def _sep(self, parent, row):
        ctk.CTkFrame(parent, height=1, fg_color=("gray80", "gray30")).grid(
            row=row, column=0, padx=20, pady=4, sticky="ew")

    def _section(self, parent, row, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=row, column=0, padx=20, pady=(8, 4), sticky="w")

    def _status(self, msg, color="gray"):
        self.status_bar.configure(text=msg, text_color=color)

    # ═══════════════════════════════════════════════════════
    # Logique métier
    # ═══════════════════════════════════════════════════════

    def _check_imports(self):
        if not IMPORTS_OK:
            self._status(f"Erreur import : {IMPORT_ERROR}", "red")
            messagebox.showerror("Dépendance manquante",
                                  f"Impossible d'importer :\n{IMPORT_ERROR}\n\n"
                                  "Vérifiez que src/model.py et src/utils.py existent.")

    # ── Chargement modèle ────────────────────────────────────
    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="Sélectionner le checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("Tous", "*.*")])
        if path:
            self.ckpt_path.set(path)

    def _load_model_thread(self):
        self.model_status.configure(text="Chargement...", text_color="orange")
        self._status("Chargement du modèle en cours...", "orange")
        self.progress.set(0.3)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            if not IMPORTS_OK:
                raise ImportError(IMPORT_ERROR)

            model_type = self.model_type.get()
            path = self.ckpt_path.get()
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fichier introuvable :\n{path}")

            # Créer le modèle selon le type sélectionné
            if model_type == "U-Net":
                model = UNet(n_channels=3, n_classes=1)
            elif model_type == "SegNet":
                model = SegNet(n_classes=1)
            elif model_type == "DeepLabV3":
                model = DeepLabV3(n_classes=1)
            else:
                raise ValueError(f"Type de modèle inconnu: {model_type}")

            model = model.to(self.device)
            ckpt  = torch.load(path, map_location=self.device)

            # Charger les poids selon le format du checkpoint
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                epoch = ckpt.get("epoch", "?")
                val_dice = ckpt.get("val_dice", 0)
                val_iou = ckpt.get("val_iou", 0)
            else:
                # Checkpoint direct (state_dict)
                model.load_state_dict(ckpt)
                epoch = "?"
                val_dice = 0
                val_iou = 0

            model.eval()
            self.model = model

            self.after(0, lambda: self.model_status.configure(
                text=f"{model_type} chargé\nEpoch {epoch}\nVal Dice : {val_dice:.4f}\nVal IoU  : {val_iou:.4f}",
                text_color="green"))
            self.after(0, lambda: self._status(
                f"Modèle {model_type} prêt — Epoch {epoch} | Dice val : {val_dice:.4f}", "green"))
            self.after(0, lambda: self.predict_btn.configure(state="normal"))
            self.after(0, lambda: self.progress.set(1.0))
            self.after(1500, lambda: self.progress.set(0))

        except Exception as e:
            self.after(0, lambda: self.model_status.configure(
                text=f"Erreur : {e}", text_color="red"))
            self.after(0, lambda: self._status(f"Erreur chargement modèle : {e}", "red"))
            self.after(0, lambda: self.progress.set(0))

    def _on_model_type_change(self, *args):
        """Met à jour le chemin du modèle quand le type change"""
        model_type = self.model_type.get()
        if model_type in MODEL_PATHS:
            self.ckpt_path.set(MODEL_PATHS[model_type])
            # Mettre à jour l'affichage du chemin
            if hasattr(self, 'ckpt_entry'):
                self.ckpt_entry.delete(0, 'end')
                self.ckpt_entry.insert(0, MODEL_PATHS[model_type])

    # ── Ouverture image ──────────────────────────────────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Ouvrir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("Tous", "*.*")])
        if not path:
            return

        self.current_img_path = path
        self.gt_mask = None
        self.pred_mask = None
        self._reset_canvases()
        self._reset_metrics()

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erreur", "Impossible de lire l'image.")
            return

        self.orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self.orig_rgb.shape[:2]

        # Prétraitement
        transform = get_transform(self.img_size_var.get())
        result = transform(image=self.orig_rgb,
                           mask=np.zeros((h, w), dtype=np.float32))
        self.tensor_img = result["image"].unsqueeze(0)

        # Affiche l'image originale
        self._show_image_on_canvas(0, self.orig_rgb)

        fname = os.path.basename(path)
        self.img_info.configure(
            text=f"{fname}\n{w}×{h} px")
        self._status(f"Image chargée : {fname} — Cliquez 'Segmenter'", "green")

    def _open_gt(self):
        path = filedialog.askopenfilename(
            title="Ouvrir le masque ground truth",
            filetypes=[("Images", "*.png *.jpg *.bmp"), ("Tous", "*.*")])
        if not path:
            return

        self.current_gt_path = path
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            messagebox.showerror("Erreur", "Impossible de lire le masque GT.")
            return

        self.gt_mask = (mask > 127).astype(np.float32)
        self._status(f"Masque GT chargé : {os.path.basename(path)}", "green")
        self.metrics_note.configure(
            text="Masque GT chargé — métriques disponibles après segmentation.")

        # Recalcule les métriques si on a déjà une prédiction
        if self.pred_mask is not None:
            self._compute_metrics()

    # ── Prédiction ───────────────────────────────────────────
    def _predict_thread(self):
        if self.model is None:
            messagebox.showwarning("Modèle", "Chargez d'abord le modèle.")
            return
        if self.tensor_img is None:
            messagebox.showwarning("Image", "Chargez d'abord une image.")
            return

        self.predict_btn.configure(state="disabled")
        self._status("Segmentation en cours...", "orange")
        self.progress.set(0.2)
        threading.Thread(target=self._predict, daemon=True).start()

    def _predict(self):
        try:
            with torch.no_grad():
                inp   = self.tensor_img.to(self.device)
                out   = self.model(inp)                        # [1,1,H,W]
                prob  = torch.sigmoid(out[0, 0]).cpu().numpy() # [H,W] float

            self.pred_prob = prob
            self.after(0, self._on_predict_done)

        except Exception as e:
            self.after(0, lambda: self._status(f"Erreur prédiction : {e}", "red"))
            self.after(0, lambda: self.predict_btn.configure(state="normal"))
            self.after(0, lambda: self.progress.set(0))

    def _on_predict_done(self):
        thr = self.threshold.get()
        self._apply_threshold(thr)
        self.predict_btn.configure(state="normal")
        self.progress.set(1.0)
        self.after(1500, lambda: self.progress.set(0))
        self._add_to_history()

    def _apply_threshold(self, thr):
        """Applique le seuil sur pred_prob et met à jour l'affichage."""
        if not hasattr(self, "pred_prob") or self.pred_prob is None:
            return

        self.pred_mask = (self.pred_prob > thr).astype(np.float32)

        # Redimensionne le masque à la taille originale pour l'affichage
        h, w = self.orig_rgb.shape[:2]
        mask_display = cv2.resize(self.pred_mask, (w, h),
                                   interpolation=cv2.INTER_NEAREST)

        # Masque en gris
        mask_gray = (mask_display * 255).astype(np.uint8)
        mask_rgb  = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
        self._show_image_on_canvas(1, mask_rgb)

        # Overlay rouge
        overlay = self.orig_rgb.copy().astype(float)
        overlay[mask_display == 1] = (
            overlay[mask_display == 1] * 0.4 +
            np.array([255, 30, 30]) * 0.6
        )
        overlay = overlay.clip(0, 255).astype(np.uint8)
        self._show_image_on_canvas(2, overlay)

        # Métriques
        if self.gt_mask is not None:
            self._compute_metrics()

        pct = self.pred_mask.sum() / self.pred_mask.size * 100
        self._status(
            f"Segmentation terminée — Seuil : {thr:.2f} | "
            f"Lésion : {pct:.1f}% de l'image", "green")

    def _on_threshold_change(self, val):
        thr = round(float(val), 2)
        self.thr_label.configure(text=f"{thr:.2f}")
        if hasattr(self, "pred_prob") and self.pred_prob is not None:
            self._apply_threshold(thr)

    # ── Métriques ────────────────────────────────────────────
    def _compute_metrics(self):
        if self.pred_prob is None or self.gt_mask is None:
            return

        thr  = self.threshold.get()
        pred = (self.pred_prob > thr).astype(np.float32)

        # Redimensionne GT à la taille du modèle
        img_sz = self.img_size_var.get()
        gt_res = cv2.resize(self.gt_mask, (img_sz, img_sz),
                             interpolation=cv2.INTER_NEAREST)

        # Aligne les tailles
        ph, pw = pred.shape
        gh, gw = gt_res.shape
        if (ph, pw) != (gh, gw):
            gt_res = cv2.resize(gt_res, (pw, ph), interpolation=cv2.INTER_NEAREST)

        # Calcul manuel Dice et IoU
        inter = (pred * gt_res).sum()
        union_d = pred.sum() + gt_res.sum()
        union_i = pred.sum() + gt_res.sum() - inter
        eps = 1e-6
        dice = (2 * inter + eps) / (union_d + eps)
        iou  = (inter + eps) / (union_i + eps)

        self.dice_val.configure(text=f"{dice:.4f}",
                                text_color="green" if dice > 0.8 else "orange")
        self.iou_val.configure(text=f"{iou:.4f}",
                               text_color="green" if iou > 0.7 else "orange")
        self.metrics_note.configure(
            text=f"Résultat avec seuil={thr:.2f}")

    def _reset_metrics(self):
        self.dice_val.configure(text="—", text_color=("black","white"))
        self.iou_val.configure(text="—", text_color=("black","white"))

    # ── Affichage images ─────────────────────────────────────
    def _show_image_on_canvas(self, idx, rgb_array):
        """Affiche un tableau numpy RGB sur le canvas idx."""
        self.canvas_imgs[idx] = rgb_array.copy()
        self._redraw_canvas(idx)

    def _redraw_canvas(self, idx):
        """Redessine le canvas idx à la taille actuelle."""
        if self.canvas_imgs[idx] is None:
            return
        c = self.canvases[idx]
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw < 10 or ch < 10:
            return

        img = self.canvas_imgs[idx]
        ih, iw = img.shape[:2]

        # Calcule le ratio pour tenir dans le canvas
        scale = min(cw / iw, ch / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        tk_img  = ImageTk.PhotoImage(pil_img)

        # Centre dans le canvas
        x = (cw - nw) // 2
        y = (ch - nh) // 2

        c.delete("all")
        c.create_image(x, y, anchor="nw", image=tk_img)
        c._tk_img = tk_img   # garde la référence

    def _reset_canvases(self):
        self.canvas_imgs = [None, None, None]
        for c in self.canvases:
            c.delete("all")
            c.create_text(10, 10, anchor="nw", text="Aucune image",
                          fill="gray", font=("Arial", 11))
        if hasattr(self, "pred_prob"):
            self.pred_prob = None

    # ── Sauvegarde ───────────────────────────────────────────
    def _save_mask(self):
        if self.pred_mask is None:
            messagebox.showwarning("Aucune prédiction", "Effectuez d'abord une segmentation.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="masque_predit.png",
            filetypes=[("PNG", "*.png"), ("Tous", "*.*")])
        if not path:
            return

        h, w = self.orig_rgb.shape[:2]
        mask_save = cv2.resize(
            (self.pred_mask * 255).astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path, mask_save)
        self._status(f"Masque sauvegardé : {os.path.basename(path)}", "green")

    def _save_overlay(self):
        if self.pred_mask is None:
            messagebox.showwarning("Aucune prédiction", "Effectuez d'abord une segmentation.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="overlay.png",
            filetypes=[("PNG", "*.png"), ("Tous", "*.*")])
        if not path:
            return

        h, w = self.orig_rgb.shape[:2]
        mask_d = cv2.resize(self.pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = self.orig_rgb.copy().astype(float)
        overlay[mask_d == 1] = overlay[mask_d == 1] * 0.4 + np.array([255, 30, 30]) * 0.6
        overlay = overlay.clip(0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        self._status(f"Overlay sauvegardé : {os.path.basename(path)}", "green")

    # ── Historique ───────────────────────────────────────────
    def _add_to_history(self):
        if not self.current_img_path:
            return

        thr  = self.threshold.get()
        pct  = self.pred_mask.sum() / self.pred_mask.size * 100 if self.pred_mask is not None else 0

        entry = {
            "date"     : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image"    : os.path.basename(self.current_img_path),
            "path"     : self.current_img_path,
            "threshold": round(thr, 2),
            "lesion_pct": round(pct, 2),
            "img_size" : self.img_size_var.get(),
        }

        # Ajoute métriques si dispo
        txt = self.dice_val.cget("text")
        if txt != "—":
            entry["dice"] = txt
            entry["iou"]  = self.iou_val.cget("text")

        self.history.insert(0, entry)
        self.history = self.history[:50]   # max 50 entrées
        self._save_history()
        self._refresh_history_ui()

    def _refresh_history_ui(self):
        for w in self.hist_rows_frame.winfo_children():
            w.destroy()

        if not self.history:
            ctk.CTkLabel(self.hist_rows_frame,
                         text="Aucune analyse effectuée.",
                         text_color="gray").grid(row=0, column=0, pady=20)
            return

        for i, entry in enumerate(self.history):
            row_frame = ctk.CTkFrame(self.hist_rows_frame, corner_radius=8,
                                      border_width=1,
                                      border_color=("gray80","gray30"))
            row_frame.grid(row=i, column=0, padx=8, pady=4, sticky="ew")
            row_frame.grid_columnconfigure(1, weight=1)

            # Numéro
            ctk.CTkLabel(row_frame,
                         text=f"#{len(self.history)-i:02d}",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="gray", width=36).grid(
                row=0, column=0, rowspan=2, padx=10, pady=8)

            # Nom fichier + date
            ctk.CTkLabel(row_frame,
                         text=entry["image"],
                         font=ctk.CTkFont(size=12, weight="bold"),
                         anchor="w").grid(row=0, column=1, padx=4, pady=(8,1), sticky="w")
            ctk.CTkLabel(row_frame,
                         text=entry["date"],
                         font=ctk.CTkFont(size=10), text_color="gray",
                         anchor="w").grid(row=1, column=1, padx=4, pady=(0,8), sticky="w")

            # Métriques à droite
            meta = f"Seuil {entry['threshold']} | Lésion {entry['lesion_pct']}%"
            if "dice" in entry:
                meta += f" | Dice {entry['dice']} | IoU {entry['iou']}"
            ctk.CTkLabel(row_frame, text=meta,
                         font=ctk.CTkFont(size=10), text_color="gray",
                         anchor="e").grid(row=0, column=2, rowspan=2, padx=12, pady=8)

            # Bouton recharger
            ctk.CTkButton(
                row_frame, text="Rouvrir", width=70, height=26,
                fg_color="transparent", border_width=1,
                command=lambda p=entry["path"]: self._reload_from_history(p)
            ).grid(row=0, column=3, rowspan=2, padx=(0,10), pady=8)

    def _reload_from_history(self, path):
        if os.path.exists(path):
            self.current_img_path = path
            self._open_image_from_path(path)
        else:
            messagebox.showwarning("Fichier introuvable",
                                    f"Le fichier n'existe plus :\n{path}")

    def _open_image_from_path(self, path):
        self.current_img_path = path
        self.gt_mask = None
        self.pred_mask = None
        self._reset_canvases()
        self._reset_metrics()

        img = cv2.imread(path)
        if img is None:
            return
        self.orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self.orig_rgb.shape[:2]
        transform = get_transform(self.img_size_var.get())
        result = transform(image=self.orig_rgb, mask=np.zeros((h, w), dtype=np.float32))
        self.tensor_img = result["image"].unsqueeze(0)
        self._show_image_on_canvas(0, self.orig_rgb)
        self.img_info.configure(text=f"{os.path.basename(path)}\n{w}×{h} px")
        self.tabs.set(TAB_RESULTS)
        self._status(f"Image rechargée : {os.path.basename(path)}", "green")

    def _clear_history(self):
        if messagebox.askyesno("Confirmer", "Effacer tout l'historique ?"):
            self.history = []
            self._save_history()
            self._refresh_history_ui()

    def _load_history(self):
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
        except Exception:
            self.history = []

    def _save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ── Thème ────────────────────────────────────────────────
    def _toggle_theme(self):
        mode = ctk.get_appearance_mode()
        ctk.set_appearance_mode("light" if mode == "Dark" else "dark")
        bg = "#1a1a2e" if ctk.get_appearance_mode() == "Dark" else "#f0f0f0"
        for c in self.canvases:
            c.configure(bg=bg)


# Lancement
if __name__ == "__main__":
    app = SegApp()
    app.mainloop()
