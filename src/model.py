"""
model.py
--------
Architecture U-Net pour la segmentation de lésions cutanées.

U-Net est composé de :
  - Un encodeur (chemin descendant) : extrait les features
  - Un décodeur (chemin ascendant)  : reconstruit le masque
  - Des skip connections            : relient encodeur ↔ décodeur
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Bloc de base : Double Convolution
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Bloc Conv → BN → ReLU → Conv → BN → ReLU
    C'est le bloc de base répété dans tout U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # Première convolution
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Deuxième convolution
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# Encodeur : un bloc descendant
# ─────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    MaxPool 2×2 (divise la résolution par 2)
    puis DoubleConv (extrait plus de features)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv   = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


# ─────────────────────────────────────────────
# Décodeur : un bloc ascendant
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    ConvTranspose2d (double la résolution)
    puis concatène avec la skip connection
    puis DoubleConv
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsampling : double la résolution spatiale
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                        kernel_size=2, stride=2)
        # Après concat : in_channels//2 + skip_channels
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        # 1. Upsample
        x = self.up(x)

        # 2. Ajuste la taille si nécessaire (cas bord)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=True)

        # 3. Concaténation avec la skip connection
        x = torch.cat([skip, x], dim=1)

        # 4. Double convolution
        x = self.conv(x)
        return x


# ─────────────────────────────────────────────
# U-Net complet
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net pour segmentation binaire (lésion vs fond).

    Paramètres
    ----------
    in_channels  : int — nombre de canaux d'entrée (3 pour RGB)
    out_channels : int — nombre de canaux de sortie (1 pour masque binaire)
    features     : list[int] — nombre de filtres à chaque niveau

    Entrée  : Tensor [B, 3, H, W]
    Sortie  : Tensor [B, 1, H, W] — probabilités (après sigmoid)
    """

    def __init__(self, in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512]):
        super().__init__()

        # ── Encodeur ──
        # Niveau 0 : pas de MaxPool au début
        self.enc0 = DoubleConv(in_channels, features[0])          # 256×256 → 64ch
        self.enc1 = EncoderBlock(features[0], features[1])        # 128×128 → 128ch
        self.enc2 = EncoderBlock(features[1], features[2])        # 64×64  → 256ch
        self.enc3 = EncoderBlock(features[2], features[3])        # 32×32  → 512ch

        # ── Bottleneck ──
        self.bottleneck = EncoderBlock(features[3], features[3]*2) # 16×16 → 1024ch

        # ── Décodeur ──
        self.dec3 = DecoderBlock(features[3]*2, features[3], features[3])   # → 512ch
        self.dec2 = DecoderBlock(features[3],   features[2], features[2])   # → 256ch
        self.dec1 = DecoderBlock(features[2],   features[1], features[1])   # → 128ch
        self.dec0 = DecoderBlock(features[1],   features[0], features[0])   # → 64ch

        # ── Couche de sortie ──
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # ── Encodeur (on sauvegarde les features pour les skip connections) ──
        s0 = self.enc0(x)       # skip 0 : [B, 64,  H,   W  ]
        s1 = self.enc1(s0)      # skip 1 : [B, 128, H/2, W/2]
        s2 = self.enc2(s1)      # skip 2 : [B, 256, H/4, W/4]
        s3 = self.enc3(s2)      # skip 3 : [B, 512, H/8, W/8]

        # ── Bottleneck ──
        b  = self.bottleneck(s3)  # [B, 1024, H/16, W/16]

        # ── Décodeur (utilise les skip connections) ──
        d3 = self.dec3(b,  s3)   # [B, 512, H/8,  W/8 ]
        d2 = self.dec2(d3, s2)   # [B, 256, H/4,  W/4 ]
        d1 = self.dec1(d2, s1)   # [B, 128, H/2,  W/2 ]
        d0 = self.dec0(d1, s0)   # [B, 64,  H,    W   ]

        # ── Couche finale ──
        out = self.final_conv(d0) # [B, 1, H, W]
        return out                # Pas de sigmoid ici → géré par la loss


# ═════════════════════════════════════════════
# SegNet — Encodeur-Décodeur avec Max-Unpooling
# ═════════════════════════════════════════════

class SegNetBlock(nn.Module):
    """Bloc SegNet : plusieurs convolutions + batch norm + relu."""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels if i==0 else out_channels, 
                                    out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SegNet(nn.Module):
    """
    SegNet pour segmentation binaire.
    
    Encodeur VGG-like + Décodeur avec Max-Unpooling (stocke les indices).
    
    Entrée  : Tensor [B, 3, H, W]
    Sortie  : Tensor [B, 1, H, W]
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # ── Encodeur ──
        self.enc1 = SegNetBlock(in_channels, 64, 2)
        self.enc2 = SegNetBlock(64, 128, 2)
        self.enc3 = SegNetBlock(128, 256, 3)
        self.enc4 = SegNetBlock(256, 512, 3)
        self.enc5 = SegNetBlock(512, 512, 3)
        
        # ── Décodeur ──
        self.dec5 = SegNetBlock(512, 512, 3)
        self.dec4 = SegNetBlock(512, 256, 3)
        self.dec3 = SegNetBlock(256, 128, 3)
        self.dec2 = SegNetBlock(128, 64, 2)
        self.dec1 = SegNetBlock(64, 64, 2)
        
        # ── MaxPool et MaxUnpool ──
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(2, 2)
        
        # ── Couche finale ──
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # ── Encodeur (stocke indices) ──
        x1 = self.enc1(x)
        x, idx1 = self.maxpool(x1)
        
        x2 = self.enc2(x)
        x, idx2 = self.maxpool(x2)
        
        x3 = self.enc3(x)
        x, idx3 = self.maxpool(x3)
        
        x4 = self.enc4(x)
        x, idx4 = self.maxpool(x4)
        
        x5 = self.enc5(x)
        x, idx5 = self.maxpool(x5)
        
        # ── Décodeur (utilise indices) ──
        x = self.maxunpool(x, idx5, output_size=x5.size())
        x = self.dec5(x)
        
        x = self.maxunpool(x, idx4, output_size=x4.size())
        x = self.dec4(x)
        
        x = self.maxunpool(x, idx3, output_size=x3.size())
        x = self.dec3(x)
        
        x = self.maxunpool(x, idx2, output_size=x2.size())
        x = self.dec2(x)
        
        x = self.maxunpool(x, idx1, output_size=x1.size())
        x = self.dec1(x)
        
        # ── Couche finale ──
        out = self.final(x)
        return out


# ═════════════════════════════════════════════
# DeepLabV3 — Atrous Spatial Pyramid Pooling
# ═════════════════════════════════════════════

class DeepLabV3(nn.Module):
    """
    DeepLabV3 avec backbone ResNet50 pré-entraîné.
    
    Utilise le module Atrous Spatial Pyramid Pooling (ASPP) pour
    capturer des features multi-échelle.
    
    Entrée  : Tensor [B, 3, H, W]
    Sortie  : Tensor [B, 1, H, W]
    """
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        import torchvision.models.segmentation as seg_models
        
        # Charger DeepLabV3 pré-entraîné
        weights = (seg_models.DeepLabV3_ResNet50_Weights.DEFAULT 
                   if pretrained else None)
        self.model = seg_models.deeplabv3_resnet50(
            weights=weights, 
            aux_loss=False
        )
        
        # Remplacer la tête pour segmentation binaire
        self.model.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # Les modèles torchvision retournent un dict
        output = self.model(x)
        
        # Extraire les logits
        if isinstance(output, dict):
            out = output['out']
        else:
            out = output
        
        # Assurer que la taille de sortie = taille d'entrée
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], 
                               mode='bilinear', align_corners=False)
        
        return out


# ─────────────────────────────────────────────
# Vérification rapide
# ─────────────────────────────────────────────

if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)   # batch test
    
    print("Test des 3 modèles de segmentation")
    print("=" * 60)
    
    # U-Net
    unet = UNet(in_channels=3, out_channels=1)
    with torch.no_grad():
        unet_out = unet(x)
    params_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print("U-Net")
    print(f"  Sortie  : {unet_out.shape}")
    print(f"  Params  : {params_unet:,}")
    print()
    
    # SegNet
    segnet = SegNet(in_channels=3, out_channels=1)
    with torch.no_grad():
        segnet_out = segnet(x)
    params_segnet = sum(p.numel() for p in segnet.parameters() if p.requires_grad)
    print("SegNet")
    print(f"  Sortie  : {segnet_out.shape}")
    print(f"  Params  : {params_segnet:,}")
    print()
    
    # DeepLabV3
    print("DeepLabV3 (chargement du backbone pré-entraîné...)")
    deeplab = DeepLabV3(in_channels=3, out_channels=1, pretrained=True)
    with torch.no_grad():
        deeplab_out = deeplab(x)
    params_deeplab = sum(p.numel() for p in deeplab.parameters() if p.requires_grad)
    print(f"  Sortie  : {deeplab_out.shape}")
    print(f"  Params  : {params_deeplab:,}")
    print()
    
    print("=" * 60)
    print("OK - Les 3 modèles sont opérationnels!")
