import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

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

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# K-Means
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# Visualisation 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
    c=labels, cmap='viridis', s=20
)
ax.set_xlabel('Poids à vide')
ax.set_ylabel('Puissance fiscale')
ax.set_zlabel('Prix véhicule')
plt.title(f"Clustering K-Means (k={k}) des véhicules")
plt.tight_layout()
plt.show()

# Analyse des clusters
print("\nAnalyse des clusters :")
print(df.groupby('Cluster')[colonnes_utiles].mean())


#violet = berlines
#vert = citadines
#bleu = SUV
#jeune = luxe