"""
evaluate.py
-----------
Évalue le meilleur modèle entraîné sur le set de test.
Génère les métriques finales et visualise les prédictions.

Usage :
    python src/evaluate.py
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model   import UNet
from src.utils   import dice_score, iou_score, visualize_predictions

# ── U-Net compatible avec le checkpoint sauvegardé ──
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNetNotebook(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = self.ups[i+1](torch.cat([skip, x], dim=1))
        return self.final(x)


CONFIG = {
    "images_dir"    : "data/Images",
    "masks_dir"     : "data/Masques",
    "checkpoint_path": "models/best_unet_isic.pth",
    "img_size"       : 128,   # même taille que train.py
    "batch_size"     : 4,     # CPU
    "num_workers"    : 0,     # Windows CPU
    "seed"           : 42,
}


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print("  Évaluation finale — Test Set")
    print(f"{'='*50}")
    print(f"  Device : {device}")

    # ── DataLoaders ──
    _, _, test_loader = get_dataloaders(
        images_dir  = CONFIG["images_dir"],
        masks_dir   = CONFIG["masks_dir"],
        img_size    = CONFIG["img_size"],
        batch_size  = CONFIG["batch_size"],
        seed        = CONFIG["seed"],
    )

    # ── Chargement du modèle ──
    model = UNetNotebook(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_epoch = checkpoint.get('epoch', '?')
        saved_dice  = checkpoint.get('val_dice', 0)
    else:
        # Direct state_dict
        model.load_state_dict(checkpoint)
        saved_epoch = '?'
        saved_dice  = 0
    model.eval()
    print(f"\n  Modèle chargé — Epoch {saved_epoch} | Val Dice: {saved_dice:.4f}\n")

    # ── Évaluation ──
    total_dice = 0.0
    total_iou  = 0.0
    all_images, all_masks, all_preds = [], [], []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="  Évaluation")
        for images, masks in loop:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)

            total_dice += dice_score(preds, masks)
            total_iou  += iou_score(preds, masks)

            # Stocke le premier batch pour visualisation
            if len(all_images) == 0:
                all_images = images.cpu()
                all_masks  = masks.cpu()
                all_preds  = preds.cpu()

    n = len(test_loader)
    final_dice = total_dice / n
    final_iou  = total_iou  / n

    print(f"\n{'='*50}")
    print("  RÉSULTATS FINAUX (Test Set)")
    print(f"  Dice Score : {final_dice:.4f}")
    print(f"  IoU Score  : {final_iou:.4f}")
    print(f"{'='*50}\n")

    # ── Visualisation ──
    visualize_predictions(
        all_images, all_masks, all_preds,
        n=4,
        save_path="outputs/predictions/test_predictions.png"
    )

    return final_dice, final_iou


if __name__ == "__main__":
    evaluate()
