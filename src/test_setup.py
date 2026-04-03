"""
test_setup.py
-------------
Vérifie que tout est bien configuré AVANT de lancer l'entraînement.
Lance ce script en premier : python test_setup.py
"""

import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("  TEST DE CONFIGURATION — ISIC PROJECT")
print("=" * 55)

# ── 1. Vérification des chemins ──────────────────────────
IMAGES_DIR = "data/Images"
MASKS_DIR  = "data/Masques"

print("\n[1/5] Vérification des dossiers...")
ok = True
for path, label in [(IMAGES_DIR, "Images"), (MASKS_DIR, "Masques")]:
    if os.path.isdir(path):
        files = os.listdir(path)
        print(f"  OK {label} : {path}  ({len(files)} fichiers)")
    else:
        print(f"  ✘ {label} INTROUVABLE : {path}")
        ok = False

if not ok:
    print("\n  ARRÊT : vérifie que tu es bien dans le dossier racine du projet.")
    print("  Lance : cd C:\\Users\\hp\\Desktop\\BUT\\ISIC_PROJECT  (adapte le chemin)")
    sys.exit(1)

# ── 2. Vérification d'une paire image/masque ─────────────
print("\n[2/5] Chargement d'une paire image/masque...")
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])
if not image_files:
    print("  ✘ Aucune image .jpg trouvée dans le dossier images !")
    sys.exit(1)

sample_img_name  = image_files[0]
sample_img_id    = sample_img_name.replace('.jpg', '')
sample_mask_name = f"{sample_img_id}_segmentation.png"
sample_mask_path = os.path.join(MASKS_DIR, sample_mask_name)

img  = cv2.imread(os.path.join(IMAGES_DIR, sample_img_name))
img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print(f"  ✘ Masque introuvable pour : {sample_img_name}")
    print(f"    Cherché : {sample_mask_path}")
    sys.exit(1)

print(f"  OK Image  : {sample_img_name}  shape={img.shape}")
print(f"  OK Masque : {sample_mask_name}  shape={mask.shape}")
print(f"  OK Valeurs masque : min={mask.min()}, max={mask.max()}")

# ── 3. Test DataLoader ────────────────────────────────────
print("\n[3/5] Test du DataLoader (5 images)...")
sys.path.insert(0, os.getcwd())
from src.dataset import ISICDataset, get_train_transforms

# Mini dataset de 5 images pour tester
ds = ISICDataset(IMAGES_DIR, MASKS_DIR, transform=get_train_transforms(256))
ds.image_ids = ds.image_ids[:5]

from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
images_batch, masks_batch = next(iter(loader))

print(f"  OK Batch images : {images_batch.shape}")
print(f"  OK Batch masques: {masks_batch.shape}")
print(f"  OK Images dtype : {images_batch.dtype}")
print(f"  OK Masque min/max : {masks_batch.min():.1f} / {masks_batch.max():.1f}")

# ── 4. Test modèle U-Net ──────────────────────────────────
print("\n[4/5] Test du modèle U-Net...")
from src.model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNet(in_channels=3, out_channels=1).to(device)

x   = images_batch.to(device)
with torch.no_grad():
    out = model(x)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  OK Device   : {device}")
print(f"  OK Entrée   : {x.shape}")
print(f"  OK Sortie   : {out.shape}")
print(f"  OK Paramètres : {params:,}")

# ── 5. Visualisation d'un exemple ─────────────────────────
print("\n[5/5] Génération d'une visualisation exemple...")
os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Vérification : image / masque GT / prédiction initiale", fontsize=13)

# Dénormalise l'image
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_show = (images_batch[0].cpu() * std + mean).permute(1, 2, 0).numpy().clip(0, 1)

pred_show = torch.sigmoid(out[0, 0]).cpu().numpy()
pred_bin  = (pred_show > 0.5).astype(float)

axes[0].imshow(img_show)
axes[0].set_title("Image originale")
axes[0].axis("off")

axes[1].imshow(masks_batch[0, 0].numpy(), cmap="gray")
axes[1].set_title("Masque GT")
axes[1].axis("off")

axes[2].imshow(pred_bin, cmap="gray")
axes[2].set_title("Prédiction (avant entraînement)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("outputs/test_setup_preview.png", dpi=150, bbox_inches="tight")
plt.show()
print("  OK Visualisation sauvegardée → outputs/test_setup_preview.png")

# ── Résumé ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  OK TOUT EST PRÊT — tu peux lancer l'entraînement !")
print("  Commande : python src/train.py")
print("=" * 55 + "\n")
