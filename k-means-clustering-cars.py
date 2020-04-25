# https://www.kaggle.com/skotty971/kmeans-car-clustering-unsupervised-approach
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd

dataset = pd.read_csv('data/cars.csv')
dataset.head()

# X = dataset[dataset.columns[0:-1]] # get all column names except the last one (we don't need branc)
# We have now an numpy ndarray we need to transform it into a dataframe
dataset.head()

# set correct column names (in case some has typos or extra white spaces)
dataset.columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year', 'brand']

# convert the columns to numeric
# coerce means that invalid parsing will be set as NaN
dataset = dataset[dataset.columns].apply(pd.to_numeric, errors='coerce')
cars = dataset[dataset.columns[0:-1]]
cars
sns.heatmap(cars.corr(), annot=True)  # show correlated values
# if there are null values fill the nulls with the means
cars.isnull().sum()  # count the number of null values
for i in cars.columns:
    cars[i] = cars[i].fillna(int(cars[i].mean()))
for i in cars.columns:
    print(cars[i].isnull().sum())


# using the Elbow method to find optimal number of clusters
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=100, random_state=0)
    kmeans.fit(cars)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Applying k-means to the cars dataset

kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=100,random_state=0)
y_kmeans = kmeans.fit_predict(cars)

cars = cars.values # we remove the column names, keep only the values so we get a matrix

# Visualising the clusters
plt.figure(figsize=(8, 6))  # start a new plot to draw new data
plt.scatter(cars[y_kmeans == 0, 0], cars[y_kmeans == 0, 1], s=100, c='red', label='Toyota')
plt.scatter(cars[y_kmeans == 1, 0], cars[y_kmeans == 1, 1], s=100, c='blue', label='Nissan')
plt.scatter(cars[y_kmeans == 2, 0], cars[y_kmeans == 2, 1], s=100, c='green', label='Honda')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')
plt.title('Clusters of car brands')
plt.legend()
plt.show()
