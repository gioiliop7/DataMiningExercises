import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.cluster.hierarchy as sch

n_samples = 1500
random_state = 170

np.random.seed(0)
#make and plot circles 
noisy_circles_x,noisy_circles_y = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)

plt.scatter(noisy_circles_x[:,0],noisy_circles_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

model=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
labels=model.fit_predict(noisy_circles_x)

#visualizing the clusters


plt.scatter(noisy_circles_x[labels==0, 0], noisy_circles_x[labels==0, 1], s=50, marker='o', color='red',edgecolor='black')
plt.scatter(noisy_circles_x[labels==1, 0], noisy_circles_x[labels==1, 1], s=50, marker='d', color='blue',edgecolor='black')
plt.scatter(noisy_circles_x[labels==2, 0], noisy_circles_x[labels==2, 1], s=50, marker='v', color='green',edgecolor='black')
plt.scatter(noisy_circles_x[labels==3, 0], noisy_circles_x[labels==3, 1], s=50, marker='*', color='purple',edgecolor='black')
plt.scatter(noisy_circles_x[labels==4, 0], noisy_circles_x[labels==4, 1], s=50, marker='p', color='orange',edgecolor='black')
plt.show()

#make and plot moons
noisy_moons_x,noisy_moons_y = datasets.make_moons(n_samples=n_samples, noise=.05)

plt.scatter(noisy_moons_x[:,0],noisy_moons_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#dendrogram=sch.dendrogram(sch.linkage(noisy_moons_x,method='ward'))
model=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
labels=model.fit_predict(noisy_moons_x)

#visualizing the clusters
plt.scatter(noisy_moons_x[labels==0, 0], noisy_moons_x[labels==0, 1], s=50, marker='o', color='red',edgecolor='black')
plt.scatter(noisy_moons_x[labels==1, 0], noisy_moons_x[labels==1, 1], s=50, marker='d', color='blue',edgecolor='black')
plt.scatter(noisy_moons_x[labels==2, 0], noisy_moons_x[labels==2, 1], s=50, marker='v', color='green',edgecolor='black')
plt.scatter(noisy_moons_x[labels==3, 0], noisy_moons_x[labels==3, 1], s=50, marker='*', color='purple',edgecolor='black')
plt.scatter(noisy_moons_x[labels==4, 0], noisy_moons_x[labels==4, 1], s=50, marker='p', color='orange',edgecolor='black')
plt.legend()
plt.show()


#make and plot blobs
blobs_x,blobs_y = datasets.make_blobs(n_samples=n_samples, random_state=8)

plt.scatter(blobs_x[:,0],blobs_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#dendrogram=sch.dendrogram(sch.linkage(blobs_x,method='ward'))
model=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
labels=model.fit_predict(blobs_x)

#visualizing the clusters
plt.scatter(blobs_x[labels==0, 0], blobs_x[labels==0, 1], s=50, marker='o', color='red',edgecolor='black')
plt.scatter(blobs_x[labels==1, 0], blobs_x[labels==1, 1], s=50, marker='d', color='blue',edgecolor='black')
plt.scatter(blobs_x[labels==2, 0], blobs_x[labels==2, 1], s=50, marker='v', color='green',edgecolor='black')
plt.scatter(blobs_x[labels==3, 0], blobs_x[labels==3, 1], s=50, marker='*', color='purple',edgecolor='black')
plt.scatter(blobs_x[labels==4, 0], blobs_x[labels==4, 1], s=50, marker='p', color='orange',edgecolor='black')
plt.show()


#make and plot no structure 
no_structure_x,no_structure_y = np.random.rand(n_samples, 2), None

plt.scatter(no_structure_x[:,0],no_structure_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#dendrogram=sch.dendrogram(sch.linkage(no_structure_x,method='ward'))
model=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
labels=model.fit_predict(no_structure_x)

#visualizing the clusters
plt.scatter(no_structure_x[labels==0, 0], no_structure_x[labels==0, 1], s=50, marker='o', color='red',edgecolor='black')
plt.scatter(no_structure_x[labels==1, 0], no_structure_x[labels==1, 1], s=50, marker='d', color='blue',edgecolor='black')
plt.scatter(no_structure_x[labels==2, 0], no_structure_x[labels==2, 1], s=50, marker='v', color='green',edgecolor='black')
plt.scatter(no_structure_x[labels==3, 0], no_structure_x[labels==3, 1], s=50, marker='*', color='purple',edgecolor='black')
plt.scatter(no_structure_x[labels==4, 0], no_structure_x[labels==4, 1], s=50, marker='p', color='orange',edgecolor='black')
plt.show()



# blobs with varied variances
varied_x,varied_y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)

plt.scatter(varied_x[:,0],varied_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#dendrogram=sch.dendrogram(sch.linkage(varied_x,method='ward'))
model=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
labels=model.fit_predict(varied_x)

#visualizing the clusters
plt.scatter(varied_x[labels==0, 0], varied_x[labels==0, 1], s=50, marker='o', color='red',edgecolor='black')
plt.scatter(varied_x[labels==1, 0], varied_x[labels==1, 1], s=50, marker='d', color='blue',edgecolor='black')
plt.scatter(varied_x[labels==2, 0], varied_x[labels==2, 1], s=50, marker='v', color='green',edgecolor='black')
plt.scatter(varied_x[labels==3, 0], varied_x[labels==3, 1], s=50, marker='*', color='purple',edgecolor='black')
plt.scatter(varied_x[labels==4, 0], varied_x[labels==4, 1], s=50, marker='p', color='orange',edgecolor='black')
plt.show()


# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples,random_state = random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

#plot
plt.scatter(X_aniso[:,0],X_aniso[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#dendrogram=sch.dendrogram(sch.linkage(X_aniso,method='ward'))
model=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
labels=model.fit_predict(X_aniso)

#visualizing the clusters
plt.scatter(X_aniso[labels==0, 0], X_aniso[labels==0, 1], s=50, marker='o', color='red',edgecolor='black')
plt.scatter(X_aniso[labels==1, 0], X_aniso[labels==1, 1], s=50, marker='d', color='blue',edgecolor='black')
plt.scatter(X_aniso[labels==2, 0], X_aniso[labels==2, 1], s=50, marker='v', color='green',edgecolor='black')
plt.scatter(X_aniso[labels==3, 0], X_aniso[labels==3, 1], s=50, marker='*', color='purple',edgecolor='black')
plt.scatter(X_aniso[labels==4, 0], X_aniso[labels==4, 1], s=50, marker='p', color='orange',edgecolor='black')
plt.show()