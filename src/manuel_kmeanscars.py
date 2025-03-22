import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# Charger le fichier CSV
DATA_PATH = os.path.join("..", "data", "ADEME-CarLabelling.csv")
df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')

# Sélection des colonnes utiles
colonnes_utiles = ['Poids à vide', 'Puissance fiscale', 'Prix véhicule']
df = df[colonnes_utiles].dropna()

# Nettoyage des valeurs aberrantes
df = df[(df['Poids à vide'] > 400) & 
        (df['Puissance fiscale'] > 1) & 
        (df['Prix véhicule'] > 1000)]

# Multiplier artificiellement les données pour tester la scalabilité (réduit x10 pour accélérer)
df = pd.concat([df] * 10, ignore_index=True)

# Convertir en tableau numpy
X = df.values

# Normalisation manuelle (centrer-réduire)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Chronomètre début
start_time = time.time()

# K-Means manuel (sans sklearn)
k = 4
np.random.seed(42)
centroids = X_scaled[np.random.choice(X_scaled.shape[0], k, replace=False)]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(data, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append(data[np.random.randint(0, data.shape[0])])
    return np.array(new_centroids)

# Boucle principale
max_iter = 100
for _ in range(max_iter):
    labels = assign_clusters(X_scaled, centroids)
    new_centroids = update_centroids(X_scaled, labels, k)
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

# Chronomètre fin
end_time = time.time()
print(f"\nTemps d'exécution total : {end_time - start_time:.2f} secondes")

# Visualisation 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
    c=labels, cmap='viridis', s=1  # point size ajusté pour grande densité
)
ax.set_xlabel('Poids à vide')
ax.set_ylabel('Puissance fiscale')
ax.set_zlabel('Prix véhicule')
plt.title(f"Clustering K-Means (k={k}) des véhicules (version manuelle)")
plt.tight_layout()
plt.show()

# Analyse des clusters
df['Cluster'] = labels
print("\nAnalyse des clusters :")
print(df.groupby('Cluster')[colonnes_utiles].mean())