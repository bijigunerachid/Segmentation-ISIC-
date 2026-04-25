"""
train_deeplabv3.py
------------------
Script d'entrainement du DeepLabV3 (ResNet-50) sur ISIC 2018 Task 1.

Usage :
    python src/train_deeplabv3.py

Le meilleur modele (selon val Dice) est sauvegarde dans outputs/deeplabv3/.

Notes :
- Le backbone ResNet-50 est initialise avec les poids ImageNet.
- Un learning rate plus faible est applique au backbone (fine-tuning).
- Recommande sur GPU (Google Colab ou CUDA local).
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as seg_models
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.utils   import DiceBCELoss, dice_score, iou_score, plot_training_curves


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG = {
    "images_dir"    : "data/Images",
    "masks_dir"     : "data/Masques",
    "output_dir"    : "outputs/deeplabv3",

    "img_size"      : 256,
    "batch_size"    : 8,
    "num_epochs"    : 47,
    "learning_rate" : 1e-4,     # tete de classification
    "backbone_lr"   : 1e-5,     # backbone ResNet-50 (fine-tuning lent)
    "weight_decay"  : 1e-5,

    "val_split"     : 0.15,
    "test_split"    : 0.15,
    "num_workers"   : 0,
    "seed"          : 42,
    "threshold"     : 0.5,
}


# ─────────────────────────────────────────────
# Construction du modele
# ─────────────────────────────────────────────

def build_model():
    """
    DeepLabV3 ResNet-50 pre-entraine ImageNet.
    La tete classifier[4] est remplacee pour la segmentation binaire.
    aux_loss=True est requis car les poids pre-entraines incluent
    la branche auxiliaire.
    """
    weights = seg_models.DeepLabV3_ResNet50_Weights.DEFAULT
    model   = seg_models.deeplabv3_resnet50(weights=weights, aux_loss=True)

    # Remplacer la tete principale : 256 -> 1
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    # Remplacer la tete auxiliaire : 256 -> 1
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    return model


def get_optimizer(model):
    """
    Deux groupes de parametres :
    - backbone (ResNet-50) : lr faible pour ne pas detruire les features pre-entrainées
    - tetes ASPP + classifier : lr normal
    """
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.classifier.parameters()) +
                       list(model.aux_classifier.parameters()))

    return optim.Adam([
        {"params": backbone_params, "lr": CONFIG["backbone_lr"]},
        {"params": head_params,     "lr": CONFIG["learning_rate"]},
    ], weight_decay=CONFIG["weight_decay"])


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
        output = model(images)

        # Sortie principale
        preds = output["out"]
        if preds.shape[-2:] != masks.shape[-2:]:
            import torch.nn.functional as F
            preds = F.interpolate(preds, size=masks.shape[-2:],
                                  mode="bilinear", align_corners=False)

        loss = criterion(preds, masks)

        # Perte auxiliaire (poids 0.4 comme dans la litterature)
        if "aux" in output:
            aux = output["aux"]
            if aux.shape[-2:] != masks.shape[-2:]:
                import torch.nn.functional as F
                aux = F.interpolate(aux, size=masks.shape[-2:],
                                    mode="bilinear", align_corners=False)
            loss = loss + 0.4 * criterion(aux, masks)

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

            output = model(images)
            preds  = output["out"]

            import torch.nn.functional as F
            if preds.shape[-2:] != masks.shape[-2:]:
                preds = F.interpolate(preds, size=masks.shape[-2:],
                                      mode="bilinear", align_corners=False)

            loss = criterion(preds, masks)
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

    print(f"\n{'='*55}")
    print("  Entrainement DeepLabV3 (ResNet-50) — ISIC 2018 Task 1")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {CONFIG['num_epochs']}")
    print(f"  Img size    : {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"  Batch       : {CONFIG['batch_size']}")
    print(f"  LR head     : {CONFIG['learning_rate']}")
    print(f"  LR backbone : {CONFIG['backbone_lr']}")
    print(f"{'='*55}\n")

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

    model     = build_model().to(device)
    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)
    criterion = DiceBCELoss()

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val_dice = 0.0
    ckpt_path = os.path.join(CONFIG["output_dir"], "best_deeplabv3_isic.pth")

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
                "epoch"               : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice"            : val_dice,
                "val_iou"             : val_iou,
                "config"              : CONFIG,
            }, ckpt_path)
            print(f"  Meilleur modele sauvegarde (Dice={val_dice:.4f})")

    print(f"\n{'='*55}")
    print("  Entrainement termine !")
    print(f"  Meilleur Dice Val : {best_val_dice:.4f}")
    print(f"  Checkpoint        : {ckpt_path}")
    print(f"{'='*55}\n")

    # Sauvegarde historique JSON
    metrics_path = os.path.join(CONFIG["output_dir"], "history.json")
    with open(metrics_path, "w") as f:
        json.dump({"history": history, "config": CONFIG,
                   "best_val_dice": best_val_dice}, f, indent=2)

    # Courbes d'entrainement
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_dice"], history["val_dice"],
        save_path=os.path.join(CONFIG["output_dir"], "training_curves.png")
    )


if __name__ == "__main__":
    main()
