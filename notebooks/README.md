# 📓 ISIC_Project — Notebooks Google Colab

Ce dossier contient **4 notebooks** à exécuter dans l'ordre sur Google Colab.

## 📋 Ordre d'exécution

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_Setup_et_Exploration.ipynb` | Connexion Drive, exploration des données, visualisation |
| 2 | `02_Preprocessing_et_Dataset.ipynb` | Augmentations, Dataset PyTorch, DataLoaders |
| 3 | `03_Modele_et_Entrainement.ipynb` | U-Net, entraînement, sauvegarde du modèle |
| 4 | `04_Evaluation_et_Predictions.ipynb` | Métriques, visualisation, rapport final |

## 🚀 Comment utiliser

1. Va sur https://colab.research.google.com
2. **Fichier → Importer le notebook** → charge le `.ipynb`
3. Active le GPU : **Environnement d'exécution → Modifier le type → T4 GPU**
4. Exécute les cellules dans l'ordre (Ctrl+Enter)

## 📁 Structure attendue sur Drive

```
ISIC_Project/
├── data/
│   ├── Images/     ← images de lésions (.jpg/.png)
│   └── Masques/    ← masques binaires (.jpg/.png)
├── src/            ← tes scripts Python
├── notebooks/      ← ce dossier
└── outputs/        ← résultats générés automatiquement
```

## 📦 Dépendances (installées automatiquement)
- PyTorch, torchvision
- albumentations
- opencv-python-headless
- scikit-learn, matplotlib, seaborn, tqdm
