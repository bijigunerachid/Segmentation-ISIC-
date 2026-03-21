"""
train.py
--------
Script principal d'entraînement du U-Net sur ISIC 2018 Task 1.

Usage :
    python src/train.py

Le meilleur modèle (selon val Dice) est sauvegardé dans outputs/checkpoints/.
"""

import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm

# Ajoute le dossier racine au path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model   import UNet
from src.utils   import DiceBCELoss, dice_score, iou_score, \
                         visualize_predictions, plot_training_curves


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG = {
    # Chemins
    "images_dir"    : "data/Images",
    "masks_dir"     : "data/Masques",
    "checkpoint_dir": "outputs/checkpoints",

    # ⚡ Paramètres optimisés pour CPU
    "img_size"      : 128,   # 128 au lieu de 256 → 4x plus rapide
    "batch_size"    : 4,     # 4 au lieu de 8 → moins de RAM
    "num_epochs"    : 20,    # 20 epochs suffisent pour valider
    "learning_rate" : 1e-4,
    "weight_decay"  : 1e-5,

    "val_split"     : 0.1,
    "test_split"    : 0.1,
    "num_workers"   : 0,     # 0 sur Windows CPU → évite les erreurs
    "seed"          : 42,
}


# ─────────────────────────────────────────────
# Fonctions train / validation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Entraîne le modèle sur une epoch complète."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    loop = tqdm(loader, desc="  Train", leave=False)
    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        # ── Forward ──
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)

        # ── Backward ──
        loss.backward()
        optimizer.step()

        # ── Métriques ──
        batch_dice  = dice_score(preds.detach(), masks)
        total_loss += loss.item()
        total_dice += batch_dice

        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n


def validate(model, loader, criterion, device):
    """Évalue le modèle sur le set de validation."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="  Val  ", leave=False)
        for images, masks in loop:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = criterion(preds, masks)

            total_loss += loss.item()
            total_dice += dice_score(preds, masks)
            total_iou  += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────

def main():
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Entraînement U-Net — ISIC 2018 Task 1")
    print(f"{'='*50}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {CONFIG['num_epochs']}")
    print(f"  Img size : {CONFIG['img_size']}×{CONFIG['img_size']}")
    print(f"  Batch    : {CONFIG['batch_size']}")
    print(f"{'='*50}\n")

    # ── DataLoaders ──
    train_loader, val_loader, _ = get_dataloaders(
        images_dir  = CONFIG["images_dir"],
        masks_dir   = CONFIG["masks_dir"],
        img_size    = CONFIG["img_size"],
        batch_size  = CONFIG["batch_size"],
        val_split   = CONFIG["val_split"],
        test_split  = CONFIG["test_split"],
        num_workers = CONFIG["num_workers"],
        seed        = CONFIG["seed"],
    )

    # ── Modèle ──
    model = UNet(in_channels=3, out_channels=1).to(device)

    # ── Optimizer + Scheduler ──
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=5
    )

    # ── Loss ──
    criterion = DiceBCELoss()

    # ── Historique ──
    train_losses, val_losses = [], []
    train_dices,  val_dices  = [], []
    best_val_dice = 0.0

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Boucle d'entraînement ──
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\n[Epoch {epoch:02d}/{CONFIG['num_epochs']}]")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device
        )

        # Scheduler sur le Dice de validation
        scheduler.step(val_dice)

        # Sauvegarde historique
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print(f"  Train → Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f}   | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        # Sauvegarde du meilleur modèle
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save({
                'epoch'     : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice'  : val_dice,
                'val_iou'   : val_iou,
                'config'    : CONFIG,
            }, ckpt_path)
            print(f"  ✔ Meilleur modèle sauvegardé (Dice={val_dice:.4f})")

    # ── Résumé final ──
    print(f"\n{'='*50}")
    print(f"  Entraînement terminé !")
    print(f"  Meilleur Dice Val : {best_val_dice:.4f}")
    print(f"{'='*50}\n")

    # ── Courbes d'entraînement ──
    plot_training_curves(
        train_losses, val_losses,
        train_dices,  val_dices,
        save_path="outputs/training_curves.png"
    )


if __name__ == "__main__":
    main()
