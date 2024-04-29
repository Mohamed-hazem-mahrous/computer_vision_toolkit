import numpy as np
import cv2


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
        return labels.astype(int) 


    # def Agglomerative(self,)