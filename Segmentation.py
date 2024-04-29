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
        # Initialize the output image as a black image
        lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype(np.float32)
        # image_copy = self.image.copy()
        rows, cols = lab_image.shape[:2]
        segmented = np.full(lab_image.shape, np.inf)
        # print(segmented[0, 0])

        # Create a queue to keep track of pixels to be checked
        queue = []
        queue.append(seed)

        # Define the intensity of the seed point
        seed_intensity = lab_image[seed[0], seed[1]][0]

        # Define the connectivity (8-connectivity in this case)
        connectivity = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        # Iterate through the queue until it's empty
        while queue:
            # Get the current pixel from the queue
            print("3 bbbbb")

            current_pixel = queue.pop(0)
            x, y = current_pixel

            # Check the intensity difference between the current pixel and the seed
            current_intensity = lab_image[x, y][0]
            intensity_diff = np.linalg.norm(current_intensity - seed_intensity)  # Euclidean distance

            # If the intensity difference is less than the threshold, add the pixel to the segmented region
            if intensity_diff <= threshold:
                segmented[x, y] = [255, 255, 255]  # Color the segmented region in red

                # Check the connectivity of the current pixel with its neighbors
                for dx, dy in connectivity:
                    nx, ny = x + dx, y + dy

                    # Check if the neighbor is within the image boundaries
                    if 0 <= nx < rows and 0 <= ny < cols:
                        print(segmented[nx, ny])
                        # Check if the neighbor pixel is not already segmented and add it to the queue
                        if np.array_equal(segmented[nx, ny], [np.inf, np.inf, np.inf]):
                            print("llllll")
                            queue.append((nx, ny))

            else:
                segmented[x, y] = lab_image[x, y]

        return segmented

   