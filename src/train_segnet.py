"""
train_segnet.py
---------------
Script d'entrainement du SegNet sur ISIC 2018 Task 1.

Usage :
    python src/train_segnet.py

Le meilleur modele (selon val Dice) est sauvegarde dans outputs/segnet/.
"""

import os
import sys
import json
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model   import SegNet
from src.utils   import DiceBCELoss, dice_score, iou_score, plot_training_curves


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG = {
    "images_dir"    : "data/Images",
    "masks_dir"     : "data/Masques",
    "output_dir"    : "outputs/segnet",

    "img_size"      : 256,
    "batch_size"    : 8,
    "num_epochs"    : 30,
    "learning_rate" : 1e-4,
    "weight_decay"  : 1e-5,

    "val_split"     : 0.15,
    "test_split"    : 0.15,
    "num_workers"   : 0,
    "seed"          : 42,
    "threshold"     : 0.5,
}


# ─────────────────────────────────────────────
# Train / Validation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_dice = 0.0

    loop = tqdm(loader, desc="  Train", leave=False)
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        batch_dice  = dice_score(preds.detach(), masks)
        total_loss += loss.item()
        total_dice += batch_dice
        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="  Val  ", leave=False)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss  = criterion(preds, masks)

            total_loss += loss.item()
            total_dice += dice_score(preds, masks)
            total_iou  += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*50}")
    print("  Entrainement SegNet — ISIC 2018 Task 1")
    print(f"{'='*50}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {CONFIG['num_epochs']}")
    print(f"  Img size : {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"  Batch    : {CONFIG['batch_size']}")
    print(f"{'='*50}\n")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

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

    model     = SegNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    criterion = DiceBCELoss()

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val_dice = 0.0
    ckpt_path = os.path.join(CONFIG["output_dir"], "best_segnet.pth")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\n[Epoch {epoch:02d}/{CONFIG['num_epochs']}]")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device)

        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(f"  Train -> Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}   | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch"            : epoch,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice"         : val_dice,
                "val_iou"          : val_iou,
                "config"           : CONFIG,
            }, ckpt_path)
            print(f"  Meilleur modele sauvegarde (Dice={val_dice:.4f})")

    print(f"\n{'='*50}")
    print("  Entrainement termine !")
    print(f"  Meilleur Dice Val : {best_val_dice:.4f}")
    print(f"  Checkpoint        : {ckpt_path}")
    print(f"{'='*50}\n")

    # Sauvegarde historique JSON
    metrics_path = os.path.join(CONFIG["output_dir"], "metrics_segnet.json")
    with open(metrics_path, "w") as f:
        json.dump({"history": history, "config": CONFIG,
                   "best_val_dice": best_val_dice}, f, indent=2)

    # Courbes d'entrainement
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_dice"], history["val_dice"],
        save_path=os.path.join(CONFIG["output_dir"], "training_curves_segnet.png")
    )


if __name__ == "__main__":
    main()
