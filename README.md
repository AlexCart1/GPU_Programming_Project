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
==============================
🚀 GUIDE INSTALLATION PROJET GPU K-MEANS
==============================

📁 Dossier de destination :
------------------------------
Allez dans l’explorateur Windows :
1. Ouvrir le chemin : C:\Users\<votre_nom_utilisateur>\
2. Créer un dossier appelé : GitRepos
3. Ouvrir un terminal (PowerShell ou CMD) dans ce dossier
   → Clique droit dans le dossier > "Ouvrir dans le terminal"

📥 Cloner le projet Git :
------------------------------
Dans le terminal, tapez la commande suivante :
git clone <URL_DU_REPO_GIT>

Exemple :
git clone https://github.com/votre-repo/projet-gpu-kmeans.git

Puis entrez dans le dossier cloné :
cd projet-gpu-kmeans

🐍 Créer l'environnement Python virtuel :
------------------------------
python -m venv .venv

🎯 Activer l’environnement :
- Sur Windows :
  .venv\Scripts\activate

- Sur Linux/macOS :
  source .venv/bin/activate

📦 Installer les dépendances :
------------------------------
Assurez-vous que le fichier `requirements.txt` est présent, puis tapez :

pip install -r requirements.txt

📂 Lancer le projet avec VS Code :
------------------------------
1. Si vous avez VS Code installé, tapez dans le terminal :

code .

2. Ouvrez ensuite le fichier à tester :
- Exemple : `src/kmeans_sklearn.py`

3. Cliquez sur ▶️ en haut à droite pour exécuter le code.

==============================
📌 Remarques :
- Le dossier `data/` contient les fichiers CSV (dataset capteurs ou image RGB).
- Le script `kmeans_sklearn.py` applique K-Means avec scikit-learn.
- L’environnement `.venv` contient tous les packages nécessaires.
==============================

💬 En cas de souci, pensez à :
- Vérifier que Python est bien installé (`python --version`)
- Vérifier que Git est installé (`git --version`)
- Demander à [Nom du référent technique] si besoin

