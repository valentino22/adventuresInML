import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#  create some random data
x = np.array([1, 5, 1.5, 8, 1, 9])
y = np.array([2, 8, 1.8, 8, 0.6, 11])

# plot the data to get visual information and to decide how many clusters I need
# plt.scatter(x,y)
# plt.show()

# match the x and y values [[1,2], [5, 8] , ...]
X = np.stack((x,y), axis=1)
print(X)

# if we plot the data first it gives a good indication on how many clusters I'll need
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_ # get the coordinates of the cluster centers
labels = kmeans.labels_

print(centroids) # [[1.16666667 1.46666667] [7.33333333 9.]]
# for each coordinate we get the categorization, eg.: first item (1,2) belongs to category 0
# second item (5, 8) to category 1
# third item (1.5, 1.8) to category 0, etc.
# I can use this information to plot the items with different colours
print(labels) # [0 1 0 1 0 1]

# plot the different categoried with different color
# using green and red for the 2 clusters, the v and ^ is to use triangle marker
colors = ["gv", "r^"]
for i in range(len(X)):
    print("coordinate", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

# plot the centroids
plt.scatter(centroids[:,0] , centroids[:,1], marker='x', s=150, linewidths=5, zorder = 10 )
plt.show()

# all rows, but only first column, in this case it's going to be the values on x coordinate [1.16666667, 7.33333333]
# centroids[:,0]

# all rows, but only second column, in this case it's going to be the values on y coordinate [1.46666667, 9.]
# centroids[:,1]