import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, DBSCAN
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Load the dataset
D = np.load("D.npy")
r = np.load("r.npy")
n_samples, n_features = D.shape

# Split into different cluster with K-means
n_clusters = 10
model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
model.fit(D)
labels = model.labels_

# Visualize clusters in a plot
x = r[:, 0]
y = r[:, 1]
colours = cm.get_cmap('viridis', n_clusters)
classes = ["Cluster {}".format(i) for i in range(n_clusters)]
fig, ax = plt.subplots(1,1, figsize=(10, 6))
ax.set_xlabel("x (Å)")
ax.set_ylabel("y (Å)")
ax.axis('equal')
scatter = ax.scatter(x, y, c=labels, cmap=colours, s=15)
ax.legend(handles=scatter.legend_elements()[0], labels=classes)
fig.tight_layout()
plt.show()
