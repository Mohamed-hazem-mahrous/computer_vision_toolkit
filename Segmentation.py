import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class ImageSegmentation:
    def __init__(self, image):
        self.image = image

    def kmeans_segmentation(self, k, max_iter=100, threshold=1e-5):
        # Convert the image into a numpy array,
        img = np.array(self.image)  # 3d array

        # Reshape the numpy array into a 2D array, where each row is a pixel and each column is a color channel
        img_shape = img.shape
        img_2d = img.reshape(img_shape[0] * img_shape[1], img_shape[2])

        # Initialize k centroids randomly
        # img_2d.shape[0]: the number of points , k: no. of centroids, replace=False means no duplicate centroids
        centroids_idx = np.random.choice(img_2d.shape[0], k, replace=False)
        centroids = img_2d[centroids_idx]

        # Initialize the labels array
        labels = np.zeros(img_2d.shape[0], dtype=int)
        distances = np.full((img_2d.shape[0], k), np.inf)  # list of distances from each centroid for each point to choose minimum

        # Run the algorithm for max_iter iterations
        for _ in range(max_iter):
            distances[:] = np.sqrt(np.sum((img_2d[:, np.newaxis] - centroids) ** 2, axis=2))

            new_labels = np.zeros(img_2d.shape[0],dtype=int)

            new_labels = np.argmin(distances, axis=1)

            # Check if the labels have changed
            if np.array_equal(new_labels, labels):
                break

            # # Check if the difference between the old and new centroids is less than the threshold value
            if np.sum(np.abs(centroids - np.array(
                    [np.mean(img_2d[new_labels == i], axis=0) for i in range(k)]))) < threshold:
                break

            labels = new_labels  # update the labels
            centroids = np.array([np.mean(img_2d[labels == i], axis=0) for i in range(k)])  # update the centroids

        labels = labels.reshape(img_shape[0], img_shape[1])
        
        # Display the segmented image
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        colormap = plt.cm.viridis  # Choose a colormap
        
        # Normalize labels to [0, 1] for colormap
        normalized_labels = labels / (num_labels )
        
        # Map labels to colors
        rgb_image = colormap(normalized_labels)

        return rgb_image

    def mean_shift(self, bandwidth):
        # Load the input image in BGR format

        # Convert the input image from BGR to LAB color space
        # lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)

        # Compute the mean shift vector for each pixel
        height, width, channels = self.image.shape
        ms_image = np.zeros((height, width), dtype=np.int32) #2d array to store labels of image 
        label = 1 #initial label 
        
        #loop over each pixel in the image 
        for y in range(height):
            for x in range(width):
                #if the pixel is not labeled 
                if ms_image[y, x] == 0:
                    #initialize ms vector to check on it for convergence  
                    ms_vector = np.zeros(channels, dtype=np.float32)
                    #pixel vector containing the color of the pixel 
                    pixel_vector = self.image[y, x].astype(np.float32)
                    #initialize number of pixels in that cluster to zero 
                    count = 0
                    
                    while True:
                        prev_ms_vector = ms_vector 
                        prev_count = count
                        #loop over nerighboring pixels within the bandwidth(window)
                        for i in range(max(y - bandwidth, 0), min(y + bandwidth, height)):
                            for j in range(max(x - bandwidth, 0), min(x + bandwidth, width)):
                                #if the neighbor pixel is not labeled and the distance in color space is less than bandwidth
                                if ms_image[i, j] == 0 and np.linalg.norm(pixel_vector - self.image[i, j]) < bandwidth:
                                    ms_vector += self.image[i, j] #accumelate the color of that pixel to average the color of the cluster at the end 
                                    count += 1 #increment number of pixels in that cluster
                                    ms_image[i, j] = label #label that pixel 
                        ms_vector /= count # get the mean color of that cluster 
                        
                        #check convergence based on distance between color and count  of the cluster 
                        if np.linalg.norm(ms_vector - prev_ms_vector) < 1e-5 or count == prev_count:
                            break
                    ms_image[ms_image == label] = count # assign label of the cluster to be denisty of the cluster 
                    label += 1 #increment the label  
                    
        # Convert the mean shift labels to a color image using the LAB color space
        unique_labels = np.unique(ms_image)
        # n_clusters = len(unique_labels)
        ms_color_image = np.zeros_like(self.image)
        for i, label in enumerate(unique_labels):
            mask = ms_image == label 
            ms_color_image[mask] = np.mean(self.image[mask], axis=0)

        return ms_color_image


    def region_growing(self, seed, threshold):
        # Convert the input image to the LAB color space
        lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype(np.float32)
        rows, cols = lab_image.shape[:2]
        
        # Initialize the segmented image as all zeros
        segmented = np.zeros_like(lab_image)

        # Create a mask to keep track of visited points
        visited = np.zeros((rows, cols), dtype=bool)

        # Create a queue to keep track of pixels to be checked
        queue = []
        queue.append(seed)

        # Define the intensity of the seed point
        seed_intensity = lab_image[seed[1], seed[0]][0]

        # Define the connectivity (8-connectivity in this case)
        connectivity = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        # Iterate through the queue until it's empty
        while queue:
            # Get the current pixel from the queue
            current_pixel = queue.pop(0)
            x, y = current_pixel

            # Mark the current pixel as visited
            visited[y, x] = True

            # Check the intensity difference between the current pixel and the seed
            current_intensity = lab_image[y, x][0]
            intensity_diff = np.linalg.norm(current_intensity - seed_intensity)  # Euclidean distance

            # If the intensity difference is less than the threshold and the pixel is not already segmented
            if intensity_diff <= threshold and not segmented[y, x].any():
                # Color the segmented region
                segmented[y, x] = [255, 255, 255]  # White

                # Check the connectivity of the current pixel with its neighbors
                for dx, dy in connectivity:
                    nx, ny = x + dx, y + dy

                    # Check if the neighbor is within the image boundaries and has not been visited
                    if 0 <= nx < rows and 0 <= ny < cols and not visited[ny, nx]:
                        # Add the neighbor to the queue
                        queue.append((nx, ny))

        return segmented


def euclidean_distance(point1, point2):
    """
    Computes euclidean distance of point1 and point2.
    
    point1 and point2 are lists.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def clusters_distance(cluster1, cluster2):
    """
    Computes distance between two clusters.
    
    cluster1 and cluster2 are lists of lists of points
    """
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])
  
def clusters_distance_2(cluster1, cluster2):
    """
    Computes distance between two centroids of the two clusters
    
    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:
    def __init__(self, k = 2, initial_k = 5):
        self.k = k
        self.initial_k = initial_k
        
    def initial_clusters(self, points):
        """
        partition pixels into self.initial_k groups based on color similarity
        """
        groups = {}
        d = int(256 / (self.initial_k))
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))  
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]
        
    def fit(self, points):
        # initially, assign each point to a distinct cluster
        self.clusters_list = self.initial_clusters(points)

        while len(self.clusters_list) > self.k:

            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                 key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num
                
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)
                    


    def predict_cluster(self, point):
        """
        Find cluster number of point
        """
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]

    def predict_center(self, point):
        """
        Find center of the cluster that point belongs to
        """
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center
