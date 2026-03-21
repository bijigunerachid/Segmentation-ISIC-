# 🔬 Segmentation des Lésions Cutanées — ISIC 2018

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%202018%20Task%201-green)
![Méthode](https://img.shields.io/badge/Méthodologie-CRISP--DM-orange)
![PFE](https://img.shields.io/badge/PFE-Bac%2B3%20Big%20Data-purple)

> Projet de Fin d'Études (PFE) — Licence Professionnelle Big Data  
> École Supérieure de Technologie — Université Sultan Moulay Slimane

---

## 📋 Description

Ce projet implémente un modèle de **segmentation sémantique** des lésions cutanées pour la détection précoce du cancer de la peau. Il utilise l'architecture **U-Net** entraînée sur le dataset **ISIC 2018 Task 1** et suit la méthodologie **CRISP-DM**.

**Objectif :** Pour chaque image dermoscopique, le modèle génère un masque binaire délimitant précisément la lésion cutanée.

---

## 🏗️ Architecture

```
U-Net
├── Encodeur (4 niveaux)
│   ├── Conv Block 1 : 64 filtres  — 256×256
│   ├── Conv Block 2 : 128 filtres — 128×128
│   ├── Conv Block 3 : 256 filtres — 64×64
│   └── Conv Block 4 : 512 filtres — 32×32
├── Bottleneck       : 1024 filtres — 16×16
└── Décodeur (4 niveaux)
    ├── UpConv 4 : 512 filtres + skip connection
    ├── UpConv 3 : 256 filtres + skip connection
    ├── UpConv 2 : 128 filtres + skip connection
    └── UpConv 1 : 64  filtres + skip connection → masque 1ch
```

---

## 📊 Dataset — ISIC 2018 Task 1

| Paramètre        | Valeur                          |
|------------------|---------------------------------|
| Source           | ISIC Challenge 2018             |
| Tâche            | Lesion Boundary Segmentation    |
| Nombre d'images  | 2594 paires (image + masque)    |
| Format images    | .jpg (RGB, tailles variables)   |
| Format masques   | .png (binaire 0/255)            |
| Split            | 80% train / 10% val / 10% test  |

---

## 🔧 Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/<votre-username>/skin-lesion-segmentation.git
cd skin-lesion-segmentation
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Télécharger le dataset ISIC 2018
1. Aller sur [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)
2. Télécharger **Task 1 — Training Data** (images + masques)
3. Placer les fichiers dans :
   ```
   data/
   ├── images/   ← ISIC_xxxxxxx.jpg
   └── masks/    ← ISIC_xxxxxxx_segmentation.png
   ```

---

## 🚀 Utilisation

### Entraînement
```bash
python src/train.py
```

### Évaluation sur le test set
```bash
python src/evaluate.py
```

### Notebook interactif
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 📁 Structure du projet

```
skin-lesion-segmentation/
├── data/
│   ├── images/              ← Images dermoscopiques (.jpg)
│   └── masks/               ← Masques de segmentation (.png)
├── src/
│   ├── __init__.py
│   ├── dataset.py           ← ISICDataset + DataLoaders
│   ├── model.py             ← Architecture U-Net
│   ├── train.py             ← Boucle d'entraînement
│   ├── evaluate.py          ← Évaluation finale
│   └── utils.py             ← Métriques Dice, IoU, visualisation
├── notebooks/
│   └── exploration.ipynb    ← EDA + pipeline complet
├── outputs/
│   ├── checkpoints/         ← Meilleur modèle (.pth)
│   └── predictions/         ← Visualisations des prédictions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Hyperparamètres

| Paramètre        | Valeur   |
|------------------|----------|
| Image size       | 256×256  |
| Batch size       | 8        |
| Epochs           | 50       |
| Optimizer        | Adam     |
| Learning rate    | 1e-4     |
| Weight decay     | 1e-5     |
| Loss function    | Dice + BCE |
| Scheduler        | ReduceLROnPlateau |

---

## 📈 Résultats

| Métrique      | Score (Test Set) |
|---------------|-----------------|
| Dice Score    | —               |
| IoU Score     | —               |

> Les résultats seront mis à jour après l'entraînement complet.

---

## 🧪 Méthodologie CRISP-DM

| Phase | Description |
|-------|-------------|
| 1. Compréhension du domaine | Segmentation de lésions pour détection du mélanome |
| 2. Compréhension des données | EDA : ISIC 2018, 2594 images dermoscopiques |
| 3. Prétraitement | Resize, normalisation ImageNet, augmentation Albumentations |
| 4. Modélisation | U-Net PyTorch, skip connections, loss Dice+BCE |
| 5. Évaluation | Dice Score, IoU Score |
| 6. Déploiement | Rapport PFE, présentation soutenance |

---

## 📚 Références

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Codella, N. et al. (2018). [ISIC 2018: Skin Lesion Analysis Toward Melanoma Detection](https://arxiv.org/abs/1902.03368)

---

## 👤 Auteur

**Bijigune**  
Licence Professionnelle Big Data — Bac+3  
École Supérieure de Technologie — Université Sultan Moulay Slimane  
