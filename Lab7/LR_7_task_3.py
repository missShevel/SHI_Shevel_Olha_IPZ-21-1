import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

X = np.loadtxt('data_clustering.txt', delimiter=',')
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

cluster_centers = ms.cluster_centers_
print('Centers of clusters:\n', cluster_centers)

labels = ms.labels_
num_clusters = len(np.unique(labels))
print("Number of clusters in input data =", num_clusters)

plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker)
    cluster_centers = ms.cluster_centers_[i]
    plt.plot(cluster_centers[0], cluster_centers[1], marker='o', markerfacecolor='black', markeredgecolor='black', markersize=15)
    plt.title('кластери')
plt.show()
