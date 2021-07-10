import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


n_samples = 1500
random_state = 170

np.random.seed(0)
#make and plot circles 
noisy_circles_x,noisy_circles_y = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)

plt.scatter(noisy_circles_x[:,0],noisy_circles_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#KMeans
km=KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(noisy_circles_x)
print(y_km)


#Plots
plt.scatter(noisy_circles_x[y_km == 0 ,0],noisy_circles_x[y_km==0,1],s=50,c='lightgreen', marker='s',edgecolor='black', label='cluster 1')
plt.scatter(noisy_circles_x[y_km == 1 ,0],noisy_circles_x[y_km==1,1],s=50,c='orange', marker='o',edgecolor='black', label='cluster 2')
plt.scatter(noisy_circles_x[y_km == 2 ,0],noisy_circles_x[y_km==2,1],s=50,c='darkblue', marker='s',edgecolor='black', label='cluster 3')

#Centroid Plots
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*',c='red',edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

print("KMeans Score=" , adjusted_rand_score(noisy_circles_y,y_km.round(2)))

#make and plot moons
noisy_moons_x,noisy_moons_y = datasets.make_moons(n_samples=n_samples, noise=.05)

plt.scatter(noisy_moons_x[:,0],noisy_moons_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#KMeans
km=KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(noisy_moons_x)
print(y_km)


#Plots
plt.scatter(noisy_moons_x[y_km == 0 ,0],noisy_moons_x[y_km==0,1],s=50,c='lightgreen', marker='s',edgecolor='black', label='cluster 1')
plt.scatter(noisy_moons_x[y_km == 1 ,0],noisy_moons_x[y_km==1,1],s=50,c='orange', marker='o',edgecolor='black', label='cluster 2')
plt.scatter(noisy_moons_x[y_km == 2 ,0],noisy_moons_x[y_km==2,1],s=50,c='darkblue', marker='s',edgecolor='black', label='cluster 3')

#Centroid Plots
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*',c='red',edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

print("KMeans Score=" , adjusted_rand_score(noisy_moons_y,y_km.round(2)))

#make and plot blobs
blobs_x,blobs_y = datasets.make_blobs(n_samples=n_samples, random_state=8)

plt.scatter(blobs_x[:,0],blobs_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#KMeans
km=KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(blobs_x)
print(y_km)



#Plots
plt.scatter(blobs_x[y_km == 0 ,0],blobs_x[y_km==0,1],s=50,c='lightgreen', marker='s',edgecolor='black', label='cluster 1')
plt.scatter(blobs_x[y_km == 1 ,0],blobs_x[y_km==1,1],s=50,c='orange', marker='o',edgecolor='black', label='cluster 2')
plt.scatter(blobs_x[y_km == 2 ,0],blobs_x[y_km==2,1],s=50,c='darkblue', marker='s',edgecolor='black', label='cluster 3')

#Centroid Plots
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*',c='red',edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

print("KMeans Score=" , adjusted_rand_score(blobs_y,y_km.round(2)))

#make and plot no structure 
no_structure_x,no_structure_y = np.random.rand(n_samples, 2), None

plt.scatter(no_structure_x[:,0],no_structure_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#KMeans
km=KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(no_structure_x)
print(y_km)


#Plots
plt.scatter(no_structure_x[y_km == 0 ,0],no_structure_x[y_km==0,1],s=50,c='lightgreen', marker='s',edgecolor='black', label='cluster 1')
plt.scatter(no_structure_x[y_km == 1 ,0],no_structure_x[y_km==1,1],s=50,c='orange', marker='o',edgecolor='black', label='cluster 2')
plt.scatter(no_structure_x[y_km == 2 ,0],no_structure_x[y_km==2,1],s=50,c='darkblue', marker='s',edgecolor='black', label='cluster 3')

#Centroid Plots
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*',c='red',edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


# blobs with varied variances
varied_x,varied_y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)

plt.scatter(varied_x[:,0],varied_x[:,1],edgecolor='black')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#KMeans
km=KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(varied_x)
print(y_km)


#Plots
plt.scatter(varied_x[y_km == 0 ,0],varied_x[y_km==0,1],s=50,c='lightgreen', marker='s',edgecolor='black', label='cluster 1')
plt.scatter(varied_x[y_km == 1 ,0],varied_x[y_km==1,1],s=50,c='orange', marker='o',edgecolor='black', label='cluster 2')
plt.scatter(varied_x[y_km == 2 ,0],varied_x[y_km==2,1],s=50,c='darkblue', marker='s',edgecolor='black', label='cluster 3')

#Centroid Plots
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*',c='red',edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

print("KMeans Score=" , adjusted_rand_score(varied_y,y_km.round(2)))

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

#KMeans
km=KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(X_aniso)
print(y_km)

#Plots
plt.scatter(X_aniso[y_km == 0 ,0],X_aniso[y_km==0,1],s=50,c='lightgreen', marker='s',edgecolor='black', label='cluster 1')
plt.scatter(X_aniso[y_km == 1 ,0],X_aniso[y_km==1,1],s=50,c='orange', marker='o',edgecolor='black', label='cluster 2')
plt.scatter(X_aniso[y_km == 2 ,0],X_aniso[y_km==2,1],s=50,c='darkblue', marker='s',edgecolor='black', label='cluster 3')

#Centroid Plots
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*',c='red',edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

print("KMeans Score=" , adjusted_rand_score(y,y_km.round(2)))
