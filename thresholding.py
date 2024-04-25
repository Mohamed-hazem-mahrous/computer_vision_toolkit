import numpy as np
import cv2

def optimal_thresholding(image):
    # Set the Image center
    threshold = 128
    
    # Get the image histogram
    hist, _ = np.histogram(image, range(256))

    def optimal(hist, threshold):
        # Cut the distribution into 2 regions
        hist1 = hist[:threshold]
        hist2 = hist[threshold:]
        
        # Check if either histogram segment is empty
        if hist1.sum() == 0 or hist2.sum() == 0:
            # If one of the segments is empty, return the current threshold
            return threshold
        
        # Compute the Centroids
        m1 = (hist1 * np.arange(0, threshold)).sum() / hist1.sum()
        m2 = (hist2 * np.arange(threshold, len(hist))).sum() / hist2.sum()
        
        # Compute the new threshold
        threshold2 = int(np.round((m1 + m2) / 2))
        
        if threshold != threshold2:
            # Recursively call optimal function if threshold has changed
            return optimal(hist, threshold2)
        
        return threshold

    threshold = optimal(hist, 127)

    # Apply thresholding
    thresholded_image = np.where(image >= threshold, 255, 0).astype(np.uint8)

    return threshold, thresholded_image


def optimal_local_thresholding(image, block_size):
    height, width = image.shape
    local_thresholded_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Extract the current block
            block = image[y:y + block_size, x:x + block_size]

            # Calculate the Otsu's threshold for the block
            block_threshold = optimal_thresholding(block)[0]

            # Apply local thresholding to the block
            thresholded_block = np.where(block >= block_threshold - block_size, 255, 0)

            # Assign the block to the result image
            local_thresholded_image[y:y + block_size, x:x + block_size] = thresholded_block

    return local_thresholded_image











def otsu_thresholding(image):
    # Set total number of bins in the histogram
    bins_num = 256
    
    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)
    
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]

    # Get the binary image
    thresholded_image = np.where(image >= threshold, 255, 0).astype(np.uint8)

    return threshold, thresholded_image


def otsu_local_thresholding(image, block_size):
    height, width = image.shape
    local_thresholded_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Extract the current block
            block = image[y:y + block_size, x:x + block_size]

            # Calculate the Otsu's threshold for the block
            block_threshold = otsu_thresholding(block)[0]

            # Apply local thresholding to the block
            thresholded_block = np.where(block >= block_threshold - block_size, 255, 0)

            # Assign the block to the result image
            local_thresholded_image[y:y + block_size, x:x + block_size] = thresholded_block

    return local_thresholded_image