import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score

n_samples = 1500
random_state = 170

np.random.seed(0)

#1dataset
#make and plot circles 
noisy_circles_x,noisy_circles_y = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)

plt.scatter(noisy_circles_x[:, 0], noisy_circles_x[:, 1], s=30, c='blue', marker='o', edgecolor='black',
            label='Circles')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(noisy_circles_x)

#Clustering the data into five clusters
dbscan=DBSCAN(eps=0.123,min_samples=2,metric='euclidean')
clusters=dbscan.fit_predict(X_Scaled)

#Plot the cluster assignments

plt.scatter(noisy_circles_x[dbscan.labels_ == 0, 0], noisy_circles_x[dbscan.labels_ == 0, 1], s=30, c='orange', marker='o',
            edgecolor='black', label='Circles 1')
plt.scatter(noisy_circles_x[dbscan.labels_ == 1, 0], noisy_circles_x[dbscan.labels_ == 1, 1], s=30, c='green', marker='v',
            edgecolor='black', label='Circles 2')
plt.scatter(noisy_circles_x[dbscan.labels_ == -1, 0], noisy_circles_x[dbscan.labels_ == -1, 1], s=30, c='blue', marker='+',
            edgecolor='black', label='Outliers')

plt.legend()
plt.show()
#dbscan performance:
print("DBSCAN Score=" ,adjusted_rand_score(noisy_circles_y,clusters))

#2dataset
#make and plot moons
noisy_moons_x,noisy_moons_y = datasets.make_moons(n_samples=n_samples, noise=.05)

plt.scatter(noisy_moons_x[:, 0], noisy_moons_x[:, 1], s=30, c='blue', marker='o', edgecolor='black',
            label='Circles')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(noisy_moons_x)

#Clustering the data into five clusters
dbscan=DBSCAN(eps=0.123,min_samples=2,metric='euclidean')
clusters=dbscan.fit_predict(X_Scaled)

#Plot the cluster assignments

plt.scatter(noisy_moons_x[dbscan.labels_ == 0, 0], noisy_moons_x[dbscan.labels_ == 0, 1], s=30, c='orange', marker='o',
            edgecolor='black', label='Circles 1')
plt.scatter(noisy_moons_x[dbscan.labels_ == 1, 0], noisy_moons_x[dbscan.labels_ == 1, 1], s=30, c='green', marker='v',
            edgecolor='black', label='Circles 2')
plt.scatter(noisy_moons_x[dbscan.labels_ == -1, 0], noisy_moons_x[dbscan.labels_ == -1, 1], s=30, c='blue', marker='+',
            edgecolor='black', label='Outliers')

plt.legend()
plt.show()


#dbscan performance:
print("DBSCAN Score=" ,adjusted_rand_score(noisy_moons_y,clusters))

#3dataset
#make and plot blobs
blobs_x,blobs_y = datasets.make_blobs(n_samples=n_samples, random_state=8)

plt.scatter(blobs_x[:, 0], blobs_x[:, 1], s=30, c='blue', marker='o', edgecolor='black',
            label='Circles')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(blobs_x)

#Clustering the data into five clusters
dbscan=DBSCAN(eps=0.123,min_samples=2,metric='euclidean')
clusters=dbscan.fit_predict(X_Scaled)

#Plot the cluster assignments

plt.scatter(blobs_x[dbscan.labels_ == 0, 0], blobs_x[dbscan.labels_ == 0, 1], s=30, c='orange', marker='o',
            edgecolor='black', label='Circles 1')
plt.scatter(blobs_x[dbscan.labels_ == 1, 0], blobs_x[dbscan.labels_ == 1, 1], s=30, c='green', marker='v',
            edgecolor='black', label='Circles 2')
plt.scatter(blobs_x[dbscan.labels_ == -1, 0], blobs_x[dbscan.labels_ == -1, 1], s=30, c='blue', marker='+',
            edgecolor='black', label='Outliers')

plt.legend()
plt.show()


#dbscan performance:
print("DBSCAN Score=" ,adjusted_rand_score(blobs_y,clusters))

#4dataset
#make and plot no structure 
no_structure_x,no_structure_y = np.random.rand(n_samples, 2), None

plt.scatter(no_structure_x[:, 0], no_structure_x[:, 1], s=30, c='blue', marker='o', edgecolor='black',
            label='Circles')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(no_structure_x)

#Clustering the data into five clusters
dbscan=DBSCAN(eps=0.123,min_samples=2,metric='euclidean')
clusters=dbscan.fit_predict(X_Scaled)

#Plot the cluster assignments

plt.scatter(no_structure_x[dbscan.labels_ == 0, 0], no_structure_x[dbscan.labels_ == 0, 1], s=30, c='orange', marker='o',
            edgecolor='black', label='Circles 1')
plt.scatter(no_structure_x[dbscan.labels_ == 1, 0], no_structure_x[dbscan.labels_ == 1, 1], s=30, c='green', marker='v',
            edgecolor='black', label='Circles 2')
plt.scatter(no_structure_x[dbscan.labels_ == -1, 0], no_structure_x[dbscan.labels_ == -1, 1], s=30, c='blue', marker='+',
            edgecolor='black', label='Outliers')

plt.legend()
plt.show()
#5dataset
# blobs with varied variances
varied_x,varied_y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)

plt.scatter(varied_x[:, 0], varied_x[:, 1], s=30, c='blue', marker='o', edgecolor='black',
            label='Circles')
plt.ylabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(varied_x)

#Clustering the data into five clusters
dbscan=DBSCAN(eps=0.123,min_samples=2,metric='euclidean')
clusters=dbscan.fit_predict(X_Scaled)

#Plot the cluster assignments

plt.scatter(varied_x[dbscan.labels_ == 0, 0], varied_x[dbscan.labels_ == 0, 1], s=30, c='orange', marker='o',
            edgecolor='black', label='Circles 1')
plt.scatter(varied_x[dbscan.labels_ == 1, 0], varied_x[dbscan.labels_ == 1, 1], s=30, c='green', marker='v',
            edgecolor='black', label='Circles 2')
plt.scatter(varied_x[dbscan.labels_ == -1, 0], varied_x[dbscan.labels_ == -1, 1], s=30, c='blue', marker='+',
            edgecolor='black', label='Outliers')

plt.legend()
plt.show()


#6dataset
# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples,random_state = random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

#plot
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], s=30, c='blue', marker='o', edgecolor='black',
            label='Circles')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(X_aniso)

#Clustering the data into five clusters
dbscan=DBSCAN(eps=0.123,min_samples=2,metric='euclidean')
clusters=dbscan.fit_predict(X_Scaled)

#Plot the cluster assignments

plt.scatter(X_aniso[dbscan.labels_ == 0, 0], X_aniso[dbscan.labels_ == 0, 1], s=30, c='orange', marker='o',
            edgecolor='black', label='Circles 1')
plt.scatter(X_aniso[dbscan.labels_ == 1, 0], X_aniso[dbscan.labels_ == 1, 1], s=30, c='green', marker='v',
            edgecolor='black', label='Circles 2')
plt.scatter(X_aniso[dbscan.labels_ == -1, 0], X_aniso[dbscan.labels_ == -1, 1], s=30, c='blue', marker='+',
            edgecolor='black', label='Outliers')


plt.legend()
plt.show()

#dbscan performance:
print("DBSCAN Score=" ,adjusted_rand_score(y,clusters))