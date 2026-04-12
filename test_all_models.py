"""
test_all_models.py
------------------
Teste tous les modèles disponibles (U-Net, SegNet, DeepLabV3).
Vérifie le chargement, l'inférence et les métriques de base.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.utils import dice_score, iou_score

# ── Configurations ───────────────────────────────────────────
CONFIG = {
    "images_dir": "data/Images",
    "masks_dir": "data/Masques",
    "img_size": 256,
    "batch_size": 4,  # Petit batch pour test rapide
    "seed": 42,
}

# ── Chemins des modèles ───────────────────────────────────────
MODEL_PATHS = {
    'U-Net': "models/best_unet_isic.pth",
    'SegNet': "models/checkpoints/best_model.pth",
    'DeepLabV3': "outputs/deeplabv3/best_deeplabv3_isic.pth"
}

# ── Architectures (depuis app_segmentation.py) ────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
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
    def __init__(self, n_classes=1, pretrained=False):
        super().__init__()
        import torchvision.models.segmentation as seg_models
        weights = seg_models.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = seg_models.deeplabv3_resnet50(weights=weights, aux_loss=False)
        self.model.classifier[4] = nn.Conv2d(256, n_classes, 1)
    def forward(self, x):
        return self.model(x)['out']

def load_model(model_name, model_path):
    """Charge un modèle selon son type."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'U-Net':
        model = UNetNotebook(in_channels=3, out_channels=1).to(device)
    elif model_name == 'SegNet':
        model = SegNetApp(in_channels=3, out_channels=1).to(device)
    elif model_name == 'DeepLabV3':
        model = DeepLabV3Wrapper(n_classes=1, pretrained=False).to(device)
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")

    # Charger le checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"  ✅ Modèle {model_name} chargé depuis {model_path}")
    else:
        print(f"  ❌ Modèle {model_name} INTROUVABLE: {model_path}")
        return None

    model.eval()
    return model, device

def test_model(model_name, model, device, test_loader):
    """Teste un modèle sur le set de test."""
    print(f"\n  [Test {model_name}] Évaluation rapide...")

    total_dice = 0.0
    total_iou  = 0.0
    count = 0

    with torch.no_grad():
        # Tester seulement sur 5 batches pour aller vite
        for i, (images, masks) in enumerate(test_loader):
            if i >= 5:  # Limiter à 5 batches
                break

            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            total_dice += dice_score(preds, masks)
            total_iou  += iou_score(preds, masks)
            count += 1

    avg_dice = total_dice / count if count > 0 else 0
    avg_iou  = total_iou  / count if count > 0 else 0

    print(f"  Dice Score : {avg_dice:.4f}")
    print(f"  IoU Score  : {avg_iou:.4f}")
    return avg_dice, avg_iou

def main():
    print("=" * 60)
    print("  TEST DE TOUS LES MODÈLES — ISIC PROJECT")
    print("=" * 60)

    # Charger les données de test
    print("\n[1/3] Chargement des données de test...")
    _, _, test_loader = get_dataloaders(
        images_dir=CONFIG["images_dir"],
        masks_dir=CONFIG["masks_dir"],
        img_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
    )
    print(f"  ✅ Test loader: {len(test_loader)} batches")

    # Tester chaque modèle
    print("\n[2/3] Test des modèles...")
    results = {}

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n🔍 Test du modèle: {model_name}")

        # Charger le modèle
        result = load_model(model_name, model_path)
        if result is None:
            results[model_name] = {"status": "MISSING", "dice": 0, "iou": 0}
            continue

        model, device = result

        # Tester le modèle
        try:
            dice, iou = test_model(model_name, model, device, test_loader)
            results[model_name] = {"status": "OK", "dice": dice, "iou": iou}
        except Exception as e:
            print(f"  ❌ Erreur lors du test {model_name}: {e}")
            results[model_name] = {"status": "ERROR", "dice": 0, "iou": 0}

    # Résumé final
    print("\n" + "=" * 60)
    print("  RÉSULTATS FINAUX")
    print("=" * 60)

    for model_name, result in results.items():
        status = result["status"]
        dice = result["dice"]
        iou = result["iou"]

        if status == "OK":
            print(f"  ✅ {model_name:<12} | Dice: {dice:.4f} | IoU: {iou:.4f}")
        elif status == "MISSING":
            print(f"  ❌ {model_name:<12} | MODÈLE MANQUANT")
        else:
            print(f"  ⚠️  {model_name:<12} | ERREUR DE CHARGEMENT")
    # Compter les modèles fonctionnels
    working_models = sum(1 for r in results.values() if r["status"] == "OK")
    total_models = len(results)

    print(f"\n📊 {working_models}/{total_models} modèles fonctionnels")

    if working_models == total_models:
        print("🎉 TOUS LES MODÈLES SONT OPÉRATIONNELS !")
    elif working_models > 0:
        print("⚠️ Certains modèles manquent ou ont des erreurs.")
    else:
        print("❌ Aucun modèle n'est disponible.")

if __name__ == "__main__":
    main()