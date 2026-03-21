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


# ─────────────────────────────────────────────
# Vérification rapide
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model  = UNet(in_channels=3, out_channels=1)
    x      = torch.randn(2, 3, 256, 256)   # batch de 2 images 256×256
    output = model(x)
    print(f"Entrée  : {x.shape}")
    print(f"Sortie  : {output.shape}")     # → torch.Size([2, 1, 256, 256])

    # Compte les paramètres
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {params:,}")
