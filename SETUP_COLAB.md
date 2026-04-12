# Guide Complet: ISIC Project sur Google Colab + Drive

## Pourquoi utiliser Google Colab?

| Raison | Avantage |
|--------|----------|
| **GPU gratuit** | Entraîner le modèle beaucoup plus vite (10x plus rapide qu'un CPU) |
| **Pas d'installation** | Aucune dépendance à installer sur ton ordinateur |
| **Sauvegarde automatique** | Tous les résultats sauvegardés sur Google Drive |
| **Partage facile** | Partage directement le lien Colab avec ton équipe |
| **RAM illimitée** | 16GB gratuit pour les gros datasets |

---

## Étape 1: Préparer Google Drive

### 1.1 Créer le dossier racine
1. Va sur **[Google Drive](https://drive.google.com)**
2. **Clic droit → Nouveau dossier**
3. Nomme-le: `ISIC_Project`

### 1.2 Créer la structure des sous-dossiers
À l'intérieur du dossier `ISIC_Project`, crée:
```
ISIC_Project/
├── data/
│   ├── Images/           (tes images PNG/JPG)
│   └── Masques/          (tes masques binaires)
├── src/                  (sera cloné automatiquement)
├── notebooks/            (sera cloné automatiquement)
└── outputs/              (créé automatiquement)
```

### 1.3 Ajouter tes données (Images et Masques)
1. **Télécharge** tes images ISIC 2018 depuis [ISIC Challenge](https://www.isic-archive.com/)
2. **Mets les images** dans `ISIC_Project/data/Images/`
3. **Mets les masques** dans `ISIC_Project/data/Masques/`

**Conseil:** Si ton dataset est > 1GB, compresse-le en ZIP d'abord, puis extrais-le dans Colab.

---

## Étape 2: Cloner le projet depuis GitHub

### Méthode 1: Depuis Google Colab (RECOMMANDÉ)

**Ouvre le notebook `01_Setup_et_Exploration.ipynb` et exécute:**

```python
# --- Monter Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Cloner le project depuis GitHub ---
import os
os.chdir('/content/drive/My Drive/ISIC_Project')

!git clone https://github.com/TON_USERNAME/ISIC_Project.git .

# Les fichiers src/ et notebooks/ sont maintenant dans Drive!
```

### Méthode 2: Manuelle via Google Drive

1. Va sur **[GitHub du projet](https://github.com/TON_USERNAME/ISIC_Project)**
2. **Code → Download ZIP**
3. **Unzip** et mets le contenu dans `ISIC_Project/` sur Drive

---

## Étape 3: Exécuter les Notebooks dans l'ordre

### Commençons! 

#### **Notebook 1: Setup et Exploration** (30 min)
```
notebooks/unet/01_Setup_et_Exploration.ipynb
```
**Ce qu'il fait:**
- Connecte Google Drive à Colab
- Charge les images et masques
- Montre des exemples visuels
- Vérifie que les dépendances sont ok

**À faire:**
1. Ouvre le notebook dans Colab
2. Exécute chaque cellule (Ctrl+Enter ou bouton Play)
3. Laisse le notebook tourner

---

#### **Notebook 2: Preprocessing et Dataset** (20 min)
```
notebooks/unet/02_Preprocessing_et_Dataset.ipynb
```
**Ce qu'il fait:**
- Augmente les données (rotations, flips, etc.)
- Crée un Dataset PyTorch
- Divise en train/val/test

**À faire:**
1. Exécute après le notebook 1
2. Les données prétraitées sont sauvegardées localement

---

#### **Notebook 3: Modèle et Entraînement** (2-4 heures)
```
notebooks/unet/03_Modele_et_Entrainement.ipynb
```
**Ce qu'il fait:**
- Construit l'architecture U-Net
- Entraîne le modèle sur GPU
- Sauvegarde le meilleur modèle

**À faire:**
1. Exécute après le notebook 2
2. **Laisse tourner!** (~2-4 heures avec GPU T4)
3. Le modèle est sauvegardé dans `outputs/checkpoints/`

**Conseil:** Utilise Colab sur ton téléphone pendant que ça tourne!

---

#### **Notebook 4: Évaluation et Prédictions** (30 min)
```
notebooks/unet/04_Evaluation_et_Predictions.ipynb
```
**Ce qu'il fait:**
- Évalue les métriques (Dice, IoU, Précision)
- Fait des prédictions sur de nouvelles images
- Génère des rapports

**À faire:**
1. Exécute après le notebook 3
2. Récupère les résultats depuis `outputs/`

---

## Configuration GPU (IMPORTANT!)

**Avant de lancer l'entraînement:**

1. En haut du notebook → **Environnement d'exécution**
2. **Modifier le type d'environnement**
3. Sélectionne: **GPU → T4**
4. **Enregistre**

```
L'entraînement sans GPU: 30+ heures
L'entraînement avec GPU: 2-4 heures
```

---

## Où trouver les résultats?

Après avoir exécuté tous les notebooks:

```
ISIC_Project/
└── outputs/
    ├── checkpoints/
    ├── best_model.pth          (Meilleur modèle)
    │   └── checkpoint_last.pth     (Dernier état)
    ├── history.json                (Courbes d'entraînement)
    ├── test_metrics.json           (Métriques finales)
    └── predictions/                (Images prédites)
```

**Récupère ces fichiers:**
1. Sur Google Drive → Télécharge-les
2. Ou reste sur Drive pour ultérieures analyses

---

## Dépannage

### "ModuleNotFoundError: No module named 'torch'"
```python
# Exécute au démarrage:
!pip install torch torchvision albumentations opencv-python scikit-learn tqdm
```

### "Permission denied" ou "File not found"
```python
# Vérifie le chemin:
!ls /content/drive/My\ Drive/ISIC_Project/
```

### Le GPU n'est pas utilisé
```python
import torch
print(torch.cuda.is_available())  # Doit afficher: True
print(torch.cuda.get_device_name(0))  # Doit afficher: Tesla T4
```

### Le notebook se déconnecte (après 12h)
- **Solution:** Réconnecte Drive et relance
- **Conseil:** Sauvegarde les checkpoints fréquemment

---

## Sauvegarder sur Google Drive

**Les notebooks sauvegardent automatiquement dans:**
```
/content/drive/My Drive/ISIC_Project/outputs/
```

**Mais tu peux aussi sauvegarder manuellement:**
```python
import shutil
shutil.copy(
    'outputs/checkpoints/best_model.pth',
    '/content/drive/My Drive/ISIC_Project/models/'
)
```

---

## Résumé du workflow

```
1. Préparer Drive              (5 min)
2. Cloner le projet            (2 min)
3. Importer les données        (10 min)
4. Notebook 1: Setup           (30 min)
5. Notebook 2: Preprocessing   (20 min)
6. Notebook 3: Entraînement    2-4 HEURES (GPU)
7. Notebook 4: Évaluation      (30 min)
8. Récupérer les résultats sur Drive
```

---

## Besoin d'aide?

Si tu as un problème:
1. **Vérifie le chemin** des dossiers sur Drive
2. **Redémarre le runtime** (Environnement d'exécution → Redémarrer)
3. **Regarde la console** des erreurs (Output)
4. **Réinstalle les dépendances** si nécessaire

---

## Optimisations avancées

### Batch Size plus grand (plus rapide)
```python
CONFIG['batch_size'] = 16  # Au lieu de 4
# Attention: Si out of memory, réduis
```

### Résolution plus haute (meilleure qualité)
```python
CONFIG['img_size'] = 256  # Au lieu de 128
# Attention: Plus lent et plus de RAM
```

### Nombre d'epochs plus important
```python
CONFIG['num_epochs'] = 50  # Au lieu de 20
# Plus d'entraînement = meilleur modèle (généralement)
```

---

**Bon entraînement!**
