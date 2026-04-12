"""
dataset.py
----------
Classe PyTorch Dataset pour ISIC 2018 Task 1.
Charge les paires (image, masque), applique resize + normalisation + augmentation.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=256):
    """Augmentation pour l'entraînement."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=256):
    """Pas d'augmentation pour validation/test."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class ISICDataset(Dataset):
    """
    Dataset ISIC 2018 Task 1.

    Paramètres
    ----------
    images_dir : str  — dossier contenant les images .jpg
    masks_dir  : str  — dossier contenant les masques _segmentation.png
    transform  : Albumentations Compose — transformations à appliquer
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform

        # Récupère tous les fichiers image triés
        self.image_ids = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        print(f"[Dataset] {len(self.image_ids)} images chargées depuis {images_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_id   = img_name.replace('.jpg', '').replace('.png', '')

        # ── Chargement image ──
        img_path = os.path.join(self.images_dir, img_name)
        image    = cv2.imread(img_path)
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── Chargement masque ──
        mask_name = f"{img_id}_segmentation.png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask      = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalise le masque → 0.0 ou 1.0
        mask = (mask > 127).astype(np.float32)

        # ── Transformations ──
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']          # Tensor [3, H, W]
            mask  = augmented['mask']           # Peut être numpy ou tensor

        # Convertir le masque en tensor si nécessaire et ajouter dimension canal → [1, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask = mask.unsqueeze(0).float()

        return image, mask


# ─────────────────────────────────────────────
# Fonction utilitaire : créer les DataLoaders
# ─────────────────────────────────────────────

def get_dataloaders(images_dir, masks_dir,
                    img_size=256, batch_size=8,
                    val_split=0.1, test_split=0.1,
                    num_workers=2, seed=42):
    """
    Crée les DataLoaders train / val / test.

    Retourne
    --------
    train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split

    # Liste de tous les ids
    all_ids = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])

    # Split train / (val+test)
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=val_split + test_split,
        random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_split / (val_split + test_split),
        random_state=seed
    )

    print(f"[Split] Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # ── Datasets ──
    def make_subset_dataset(ids, transform):
        """Sous-ensemble du dataset avec une liste d'ids donnée."""
        ds = ISICDataset(images_dir, masks_dir, transform)
        ds.image_ids = ids
        return ds

    train_ds = make_subset_dataset(train_ids, get_train_transforms(img_size))
    val_ds   = make_subset_dataset(val_ids,   get_val_transforms(img_size))
    test_ds  = make_subset_dataset(test_ids,  get_val_transforms(img_size))

    # ── DataLoaders ──
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader
