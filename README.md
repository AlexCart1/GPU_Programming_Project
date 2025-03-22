# GPU_Programming_Project
Projet de la matière GPU programming

## Objectifs
Ce projet a pour but d’implémenter et d’optimiser l’algorithme de clustering **K-Means** à l’aide de **CUDA** pour tirer parti de la **parallélisation GPU**.

##  Domaines d’application

### 1. 🎨 Segmentation d’images (RGB)
- Chaque pixel est un point 3D (R, G, B).
- K-Means regroupe les pixels par couleur dominante.
- Application : compression d’image, segmentation visuelle.

### 2. 📱 Données de capteurs de mouvement
- Chaque point est une mesure 3D (accélération X, Y, Z).
- K-Means regroupe les gestes ou mouvements similaires.
- Application : reconnaissance de gestes, suivi de mouvements.

---


## ⚙️ Plan du projet

1. **Choix du problème et justification**
2. **Implémentation en Python (scikit-learn)**
3. **Réimplémentation manuelle de l’algorithme**
4. **Implémentation CUDA**
5. **Comparaison des performances CPU vs GPU**
6. **Optimisations CUDA (mémoire partagée, etc.)**
7. **Tests en 3D et N-D**
8. **Conclusion sur la scalabilité et l’efficacité GPU**

---

### 🔹 Choix 1 – Principal : Détection de mouvements avec capteurs (3D)

#### 📌 Objectif :
Utiliser l’algorithme K-Means pour regrouper des données issues de capteurs de mouvement (accéléromètres) selon des patterns de mouvement (marche, course, arrêt, etc.).

#### 📊 Données :
- **Features** : `acc_x`, `acc_y`, `acc_z` (accélération sur les 3 axes)
- **Dataset utilisé** : [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- **Dimensions** : 3D

#### 💡 Intérêt :
- Problème courant dans les systèmes embarqués et objets connectés (IoT)
- Forte utilité dans la reconnaissance d’activités, la santé, le sport ou la surveillance
- Traitement massivement parallèle sur GPU (idéal pour CUDA)

---

### 🔹 Choix 2 – Secours : Segmentation d’image en RGB (3D)

#### 📌 Objectif :
Regrouper les pixels d’une image selon leur couleur pour simplifier visuellement l’image (ex: compression, effets artistiques, segmentation).

#### 📊 Données :
- **Features** : `R`, `G`, `B` (valeurs des canaux de couleur)
- **Dataset utilisé** : N’importe quelle image JPG/PNG (ex: image standard ou personnelle)
- **Dimensions** : 3D

#### 💡 Intérêt :
- Facile à tester et à visualiser (avant/après image)
- Exécution hautement parallélisable (des milliers de pixels)
- Bonne démonstration du clustering sur des données visuelles

---
