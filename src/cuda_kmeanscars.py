# Version CUDA (via Numba) du K-Means pour clustering de voitures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from numba import cuda, float32, int32
import math

# Charger les données
DATA_PATH = os.path.join("..", "data", "ADEME-CarLabelling.csv")
df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')

# Prétraitement
colonnes_utiles = ['Poids à vide', 'Puissance fiscale', 'Prix véhicule']
df = df[colonnes_utiles].dropna()
df = df[(df['Poids à vide'] > 400) & (df['Puissance fiscale'] > 1) & (df['Prix véhicule'] > 1000)]
df = pd.concat([df] * 100, ignore_index=True)
X = df.values.astype(np.float32)

# Normalisation
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

n_samples, n_features = X.shape
k = 4
max_iter = 100

# Initialisation aléatoire des centroïdes
np.random.seed(42)
centroids = X[np.random.choice(n_samples, k, replace=False)]

# Allocation mémoire sur GPU
d_X = cuda.to_device(X)
d_labels = cuda.device_array(n_samples, dtype=np.int32)
d_centroids = cuda.to_device(centroids)

@cuda.jit
def assign_labels(X, centroids, labels):
    i = cuda.grid(1)
    if i < X.shape[0]:
        min_dist = float('inf')
        label = -1
        for j in range(centroids.shape[0]):
            dist = 0.0
            for f in range(X.shape[1]):
                diff = X[i, f] - centroids[j, f]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                label = j
        labels[i] = label

@cuda.jit
def update_centroids(X, labels, centroids, k):
    shared_sum = cuda.shared.array((4, 3), dtype=float32)
    shared_count = cuda.shared.array(4, dtype=int32)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = bx * bw + tx

    if tx < k:
        for j in range(X.shape[1]):
            shared_sum[tx][j] = 0.0
        shared_count[tx] = 0
    cuda.syncthreads()

    if i < X.shape[0]:
        label = labels[i]
        cuda.atomic.add(shared_count, label, 1)
        for j in range(X.shape[1]):
            cuda.atomic.add(shared_sum[label], j, X[i, j])
    cuda.syncthreads()

    if tx < k and shared_count[tx] > 0:
        for j in range(X.shape[1]):
            centroids[tx, j] = shared_sum[tx][j] / shared_count[tx]

# Exécution principale
threads_per_block = 128
blocks_per_grid = math.ceil(n_samples / threads_per_block)

for _ in range(max_iter):
    assign_labels[blocks_per_grid, threads_per_block](d_X, d_centroids, d_labels)
    update_centroids[blocks_per_grid, threads_per_block](d_X, d_labels, d_centroids, k)

labels = d_labels.copy_to_host()
centroids = d_centroids.copy_to_host()

# Affichage
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=1)
ax.set_xlabel('Poids à vide')
ax.set_ylabel('Puissance fiscale')
ax.set_zlabel('Prix véhicule')
plt.title(f"K-Means CUDA (k={k}) sur les véhicules")
plt.tight_layout()
plt.show()

# Analyse
df['Cluster'] = labels
print("\nAnalyse des clusters :")
print(df.groupby('Cluster')[colonnes_utiles].mean())
