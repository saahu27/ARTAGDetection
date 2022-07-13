
import numpy as np
import matplotlib.pyplot as plt
import cv2



def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class K_Means():
    
    def __init__(self, K=4, max_iters=100):
        self.K = K
        self.max_iters = max_iters

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]


        for _ in range(self.max_iters):
            # Seperating samples to closest centroids
            self.clusters = self.Create_Cluster(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.Get_Centroids(self.clusters)
            
            # check if clusters have changed
            if self.Is_Converged(centroids_old, self.centroids):
                break

        
        return self.Get_Cluster_Labels(self.clusters)

    def Get_Cluster_Labels(self, clusters):
        # Classify samples as the index of their clusters
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def Create_Cluster(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.Nearest_Centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def Nearest_Centroid(self, sample, centroids):
        # distance of the current sample to each centroid, using in creating clusters
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def Get_Centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def Is_Converged(self, centroids_old, centroids):
        # Calculating distance between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 10

    def Get_Centroid(self):
        return self.centroids

image = cv2.imread("Q4image.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

k = K_Means(K=4, max_iters=100)  
y_pred = k.predict(pixel_values) 

centers = np.uint8(k.Get_Centroid())

y_pred = y_pred.astype(int)
np.unique(y_pred)
labels = y_pred.flatten()


labels = y_pred.flatten()
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)
plt.imshow(segmented_image)
plt.show()

