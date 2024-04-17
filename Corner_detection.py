from cv2 import imread, IMREAD_ANYCOLOR, line, circle
import cv2
import numpy as np
from math import ceil
import os
import time

def convert_to_grayscale(image):
        """
        Convert the RGB image to grayscale using NTSC formula.
        :param image: Input RGB image (numpy array).
        :return: Grayscale image (numpy array).
        """
        rgb_coefficients = [0.299, 0.587, 0.114]
        grayscale_image = np.dot(image[..., :3], rgb_coefficients)

        return grayscale_image.astype(np.uint8)

def apply_gaussian_filter(image, kernel_size = 3, sigma = 10):
        """
        Apply Gaussian filter to the image.
        :param image: Input image (numpy array).
        :param kernel_size: Size of the square kernel.
        :param sigma: Standard deviation of the Gaussian distribution.
        :return: Filtered image.
        """        
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size//2))**2 + (y-(kernel_size//2))**2) / (2*sigma**2)), (kernel_size, kernel_size))
        kernel /= np.sum(kernel)

        filtered_image = convolve(image, kernel)      
        
        return filtered_image



def convolve(image, kernel, padding_mode='reflect'):
    """
    Apply convolution between the input image and kernel.
    
    Args:
    image: Input image (numpy array).
    kernel: Convolution kernel (numpy array).
    padding_mode: The padding mode for the convolution.
    
    Returns:
    convolved_image: The resulting image after convolution.
    """
    # Define the padding mode
    pad_size = kernel.shape[0] // 2
    padded_image = np.pad(image, pad_width=pad_size, mode=padding_mode)
    
    # Initialize the convolved image with the same shape as the input image
    convolved_image = np.zeros_like(image)
    
    # Perform convolution
    for i in range(convolved_image.shape[0]):
        for j in range(convolved_image.shape[1]):
            # Extract the region of interest (ROI)
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            # Compute the convolution
            convolved_image[i, j] = np.sum(roi * kernel)
    
    return convolved_image

def sobel_edge(image, direction='both'):
    """
    Apply Sobel edge detection to the image.
    :param image: Input image (numpy array).
    :param direction: Direction of edge detection ('x', 'y', or 'both').
    :return: Edge-detected image.
    """
    if len(image.shape) == 3:  # Check if the image is multi-channel (color)
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Binarize the grayscale image
    _, image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    
    # Apply Sobel edge detection on the binary image
    sobel_kernel = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
    return apply_edge(image, sobel_kernel, direction)


def apply_edge(image, array, direction):
    if direction == 'Horizontal':
        edge = abs(convolve(image, array))
    elif direction == 'Vertical':
        edge = abs(convolve(image, array.T))
    else:
        edge_x = abs(convolve(image, array))
        edge_y = abs(convolve(image, array.T))
        edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge.astype(np.uint8)


def harris_corner_detection(image, threshold=0.01, window_size=3, k=0.04):
    # Step 1: Compute gradients
    dx = sobel_edge(image, 'Vertical')
    dy = sobel_edge(image, 'Horizontal')


    # Step 2: Compute products of gradients
    Ixx = dx**2
    Ixy = dx * dy
    Iyy = dy**2

    # Step 3: Apply Gaussian window to the products of gradients
    offset = window_size // 2
    height, width = image.shape
    corner_response = np.zeros_like(image, dtype=np.float32)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sxx = np.sum(Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1])
            Sxy = np.sum(Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1])
            Syy = np.sum(Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1])

            # Step 4: Compute the corner response function R
            det_M = Sxx * Syy - Sxy**2
            trace_M = Sxx + Syy
            R = det_M - k * trace_M**2

            corner_response[y, x] = R

    # Step 5: Non-maximum suppression
    corners = np.zeros_like(image)
    corner_response_max = np.max(corner_response)
    threshold_value = threshold * corner_response_max

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if corner_response[y, x] > threshold_value:
                corners[y, x] = 255

    return corners



def lambda_minus_corner_detection(image, threshold=0.2, window_size=3):
    # Step 1: Compute gradients using Sobel filter
    dx = sobel_edge(image, 'Vertical')
    dy = sobel_edge(image, 'Horizontal')

   
    # Step 2: Compute products of gradients
    Ixx = dx**2
    Ixy = dx * dy
    Iyy = dy**2
    
    # Step 3: Apply Gaussian filter to the products of gradients
    Ixx = apply_gaussian_filter(Ixx, kernel_size=window_size)
    Ixy = apply_gaussian_filter(Ixy, kernel_size=window_size)
    Iyy = apply_gaussian_filter(Iyy, kernel_size=window_size)

    # Step 4: Compute lambda-minus (smallest eigenvalue of the second moment matrix)
    height, width = image.shape
    corner_response = np.zeros_like(image, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Construct the second moment matrix for the current pixel
            M = np.array([[Ixx[y, x], Ixy[y, x]],
                          [Ixy[y, x], Iyy[y, x]]])
            
            # Compute eigenvalues of the matrix
            eigenvalues = np.linalg.eigvals(M)
            
            # Get the smallest eigenvalue (lambda-minus)
            lambda_minus = min(eigenvalues)
            
            # Step 5: Apply threshold to identify corners
            if lambda_minus > threshold:
                corner_response[y, x] = 255
    
    # Step 6: Non-maximum suppression to remove false positives
    corners = non_maximum_suppression(corner_response, threshold)
    
    return corners

def non_maximum_suppression(image, threshold):
    height, width = image.shape
    suppresed_image = np.zeros_like(image, dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Get the current pixel's corner response
            current_response = image[y, x]
            
            # Check if the current pixel's response is a local maximum
            if current_response > threshold:
                is_local_maximum = True
                for ny in range(-1, 2):
                    for nx in range(-1, 2):
                        if image[y + ny, x + nx] > current_response:
                            is_local_maximum = False
                            break
                    if not is_local_maximum:
                        break
                if is_local_maximum:
                    suppresed_image[y, x] = 255
    
    return suppresed_image