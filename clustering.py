import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random data for clustering
np.random.seed(0)
X = np.random.randn(100, 2) * 2 + np.array([10, 5])

# Perform K-means clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Get the cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the data points and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', color='red', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering')

# Display the plot
plt.show()
