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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model   import UNet
from src.utils   import dice_score, iou_score, visualize_predictions


CONFIG = {
    "images_dir"    : "data/Images",
    "masks_dir"     : "data/Masques",
    "checkpoint_path": "outputs/checkpoints/best_model.pth",
    "img_size"       : 128,   # même taille que train.py
    "batch_size"     : 4,     # CPU
    "num_workers"    : 0,     # Windows CPU
    "seed"           : 42,
}


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Évaluation finale — Test Set")
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
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    saved_epoch = checkpoint.get('epoch', '?')
    saved_dice  = checkpoint.get('val_dice', 0)
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
    print(f"  RÉSULTATS FINAUX (Test Set)")
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
