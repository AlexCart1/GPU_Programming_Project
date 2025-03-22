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
