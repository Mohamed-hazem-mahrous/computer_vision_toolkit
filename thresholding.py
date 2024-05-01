import numpy as np
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def optimal_thresholding(image):
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


def spectral_thresholding(img, num_bins=256, peaks_range=17, min_peak_threshold=0):
    peaks = []
    raw_hist = np.histogram(img, bins=num_bins)[0]

    hist = cv2.blur(raw_hist, (10, 10))
    hist_flat = hist.flatten()

    peaks, _ = find_peaks(hist_flat, distance=peaks_range)
    peaks = [(i, hist_flat[i]) for i in peaks if hist_flat[i] > max(hist_flat) * min_peak_threshold]
    # print(len(peaks))

    threshold_list = []
    for i in range(len(peaks) - 1):
        current_peak = peaks[i]
        next_peak = peaks[i + 1]

        valley_index = np.argmin(hist[current_peak[0]:next_peak[0]]) + current_peak[0]

        if hist[valley_index] < current_peak[1] and hist[valley_index] < next_peak[1]:
            threshold_list.append(valley_index)

    segmented_img = np.zeros_like(img)
    for i in range(len(threshold_list)):
        segmented_img[(img >= threshold_list[i]) & (img < (threshold_list[i + 1] if i < len(threshold_list) - 1 else 256))] = i + 1

    return segmented_img


def spectral_local_thresholding(image, block_size, peaks_range=17, min_peak_threshold=0.2):
    height, width = image.shape
    local_thresholded_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y + block_size, x:x + block_size]

            local_thresholded_image[y:y + block_size, x:x + block_size] = spectral_thresholding(block, peaks_range=peaks_range, min_peak_threshold=min_peak_threshold)

    return local_thresholded_image


def multilevel_spectral_thresholding(image, num_classes=4):
    histogram = np.histogram(image, bins=256)[0]
    normalized_histogram = histogram.ravel() / histogram.sum()

    cumulative_sum = np.cumsum(normalized_histogram)

    thresholds = np.zeros(num_classes - 1)

    def calculate_threshold(class_index, normalized_histogram, cumulative_sum):
        max_variance, best_threshold = 0, 0
        for t in range(class_index * 256 // num_classes, (class_index + 1) * 256 // num_classes):
            weight_0 = cumulative_sum[t] if t > 0 else 0
            weight_1 = cumulative_sum[-1] - weight_0

            if (weight_0 + weight_1) < 1e-5:
                continue

            mean_0 = np.sum(np.arange(0, t + 1) * normalized_histogram[:t + 1]) / weight_0
            mean_1 = np.sum(np.arange(t + 1, 256) * normalized_histogram[t + 1:]) / weight_1

            variance = weight_0 * weight_1 * ((mean_0 - mean_1) ** 2)

            if variance > max_variance:
                max_variance = variance
                best_threshold = t

        return best_threshold

    for i in range(num_classes - 1):
        thresholds[i] = calculate_threshold(i, normalized_histogram, cumulative_sum)

    segmented_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(num_classes):
        if i == 0:
            segmented_image[image <= thresholds[0]] = i
        elif i == num_classes - 1:
            segmented_image[image > thresholds[i - 1]] = i
        else:
            segmented_image[(image > thresholds[i - 1]) & (image <= thresholds[i])] = i

    return segmented_image

def multilevel_spectral_local_thresholding(image, block_size):
    height, width = image.shape
    local_thresholded_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y + block_size, x:x + block_size]

            local_thresholded_image[y:y + block_size, x:x + block_size] = multilevel_spectral_thresholding(block)

    return local_thresholded_image
