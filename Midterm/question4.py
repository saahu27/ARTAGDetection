import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# read the image
image = cv2.imread('Q4image.png')

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

def cdist1(p, c):
    count = 1
    if count == 1:
        print("Hello",p,c)
    k, _ = c.shape
    dists = []
    for i in range(k):
        dists.append(np.linalg.norm(p - c[i], axis = 1))
    return np.array(dists).T

#Function to implement steps given in previous section
def kmeans(x, k, no_of_iterations):
    
    idx = np.random.choice(len(x), k, replace = False)
    # idx = np.array([82 + 168*281, 177 + 254*281, 116 + 209*281, 256 + 214*281])
    #Randomly choosing Centroids 
    centroids = x[idx, :] #Step 1
     
    #finding the distance between centroids and all the data points
    distances = cdist1(x, centroids) #Step 2
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
     
    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids 
        
        distances = cdist1(x, centroids)
        points = np.array([np.argmin(i) for i in distances])
         
    return centroids, points

# # number of clusters (K)
k = 4
centers, labels = kmeans(pixel_values, k, 100)

# # convert back to 8 bit values
centers = np.uint8(np.round(centers))
print(centers)

segmented_image = centers[labels]
# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)

# show the image
plt.imshow(segmented_image)
plt.show()

