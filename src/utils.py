"""
utils.py
--------
Métriques d'évaluation et fonction de perte pour la segmentation binaire.
  - Dice Score
  - IoU (Intersection over Union)
  - DiceBCELoss (perte combinée)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────

def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calcule le Dice Score entre prédiction et cible.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Paramètres
    ----------
    pred      : Tensor [B, 1, H, W] — logits ou probabilités
    target    : Tensor [B, 1, H, W] — masque binaire (0 ou 1)
    threshold : float — seuil pour binariser la prédiction
    smooth    : float — évite la division par zéro

    Retourne
    --------
    float — Dice Score moyen sur le batch
    """
    pred   = torch.sigmoid(pred)
    pred   = (pred > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calcule le IoU (Jaccard Index) entre prédiction et cible.

    IoU = |A ∩ B| / |A ∪ B|

    Paramètres
    ----------
    pred      : Tensor [B, 1, H, W]
    target    : Tensor [B, 1, H, W]
    threshold : float
    smooth    : float

    Retourne
    --------
    float — IoU moyen sur le batch
    """
    pred   = torch.sigmoid(pred)
    pred   = (pred > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


# ─────────────────────────────────────────────
# Fonction de perte
# ─────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """
    Perte combinée : Dice Loss + Binary Cross-Entropy.

    Pourquoi combiner les deux ?
    - BCE  : pénalise pixel par pixel (bonne convergence initiale)
    - Dice : optimise directement la métrique d'évaluation

    Loss = BCE + Dice Loss
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth   = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # ── BCE Loss ──
        bce = self.bce_loss(pred, target)

        # ── Dice Loss ──
        pred_sig     = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(1, 2, 3))
        union        = pred_sig.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss    = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss    = dice_loss.mean()

        return bce + dice_loss


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def visualize_predictions(images, masks, preds, n=4, save_path=None):
    """
    Affiche n triplets : image | masque GT | masque prédit.

    Paramètres
    ----------
    images    : Tensor [B, 3, H, W]
    masks     : Tensor [B, 1, H, W]
    preds     : Tensor [B, 1, H, W]
    n         : int — nombre d'exemples à afficher
    save_path : str | None — chemin pour sauvegarder la figure
    """
    # Dénormalisation ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle('Image | Masque GT | Prédiction', fontsize=14)

    for i in range(min(n, images.shape[0])):
        # Dénormalise l'image
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy().clip(0, 1)

        gt_mask   = masks[i, 0].cpu().numpy()
        pred_mask = torch.sigmoid(preds[i, 0]).cpu().detach().numpy()
        pred_bin  = (pred_mask > 0.5).astype(float)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image originale')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title('Masque GT')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_bin, cmap='gray')
        axes[i, 2].set_title('Prédiction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[Visualisation] Sauvegardée → {save_path}")
    plt.show()


def plot_training_curves(train_losses, val_losses,
                          train_dices, val_dices, save_path=None):
    """
    Trace les courbes de loss et de Dice Score pendant l'entraînement.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses,   'r-o', label='Val Loss',   markersize=4)
    ax1.set_title('Courbe de Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Dice
    ax2.plot(epochs, train_dices, 'b-o', label='Train Dice', markersize=4)
    ax2.plot(epochs, val_dices,   'r-o', label='Val Dice',   markersize=4)
    ax2.set_title('Courbe de Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[Courbes] Sauvegardées → {save_path}")
    plt.show()
