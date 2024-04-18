from cv2 import imread, IMREAD_ANYCOLOR, line, circle
import cv2
import numpy as np
from math import ceil


class ImageProcessor:
    def __init__(self, filePath):
        self.filePath = filePath
        self.image = imread(self.filePath, IMREAD_ANYCOLOR)
        self.RGBhistograms, self.RGBcdf = self.get_RGB_histograms_and_cdf(self.image)
        self.image = self.convert_to_grayscale(self.image)
        self.noisy_image = None

    def get_histogram(self, image, bins_num):
        """
        Compute the histogram of the image.
        :param image: Image (numpy array).
        :param bins_num: number of bins in the histogram.
        :return: histogram of the image.
        """
        histogram = np.zeros(bins_num)
        for pixel in image:
            histogram[pixel] += 1

        return histogram

    def get_cdf(self, histogram, shape):
        """
        Compute the cumulative distribution function (CDF) of the image.
        :param histogram: histogram of the image computed from get_histogram.
        :param shape: shape of the image.
        :return: CDF of the image.
        """
        if len(shape) > 1:
            no_pixels = shape[0] * shape[1]
            prob = histogram / no_pixels
        else:
            no_pixels = shape[0]
            prob = histogram / no_pixels

        cdf = np.zeros(len(prob))
        for i in range(1, len(prob)):
            cdf[i] = cdf[i - 1] + prob[i]

        return cdf

    def histogram_equalization(self, image, max_value=255):
        """
        Apply histogram equalization to the image.
        :param image: Image (numpy array).
        :param max_value: maximum pixel value in the image.
        :return: Equalized image.
        """
        hist = self.get_histogram(image.flatten(), 256)
        cdf = self.get_cdf(hist, image.shape)
        normalize = np.rint(cdf * max_value).astype('int')

        result = normalize[image.flatten()]
        return result.reshape(image.shape)

    
    def image_normalization(self):
        # Ensure the image is in float format to handle division correctly
        image_float = self.image.astype(np.float32)
        # Compute the minimum and maximum pixel values
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        # Normalize the image to [0, 255]
        normalized_image = ((image_float - min_val) / (max_val - min_val)) * 255
        return normalized_image.astype(np.uint8)  # Convert to uint8 for QImage


    def global_thresholding(self, threshold):
        # Create an empty image for the result
        thresholded_image = np.zeros_like(self.image)

        # Apply global thresholding
        thresholded_image[self.image >= threshold] = 255

        return thresholded_image



    def local_thresholding(self, block_size, C):
        height, width = self.image.shape
        local_thresholded_image = np.zeros((height, width), dtype=np.uint8)

        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Extract the current block
                block = self.image[y:y + block_size, x:x + block_size]

                # Calculate the mean intensity of the block
                block_mean = np.mean(block)

                # Apply local thresholding to the block where if condition is true (foreground) it takes white and otherwise is black
                thresholded_block = np.where(block >= (block_mean - C), 255, 0)

                # Assign the block to the result image
                local_thresholded_image[y:y + block_size, x:x + block_size] = thresholded_block

        return local_thresholded_image
    

    def add_uniform_noise(self, image, SNR):
        """
        Add uniform noise to the image.
        :param image: Input image (numpy array).
        :param SNR: Signal-to-Noise Ratio controlling the intensity of the noise.
        :return: Noisy image.
        """
        noise = np.random.uniform(low=0, high=(SNR) * 255, size=image.shape).astype(np.uint8)
        self.noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)    
        return self.noisy_image
    

    def add_gaussian_noise(self, image, sigma):
        """
        Add Gaussian noise to the image.
        :param image: Input image (numpy array).
        :param sigma: Standard deviation of the Gaussian distribution.
        :return: Noisy image.
        """
        gaussian_noise = np.random.normal(0, sigma * 255 , image.shape).astype(np.uint8)
        self.noisy_image = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)
        return self.noisy_image

    
    def add_salt_and_pepper_noise(self, image, amount):
        """
        Add salt-and-pepper noise to the image.
        :param image: Input image (numpy array).
        :param amount: Probability of salt and pepper noise.
        :return: Noisy image.
        """
        self.noisy_image = np.copy(image)
        num_salt = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        self.noisy_image[coords[0], coords[1]] = 255

        num_pepper = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        self.noisy_image[coords[0], coords[1]] = 0
        return self.noisy_image
    







    def apply_average_filter(self, image, kernel_size=3):
        """
        Apply average filter to the image.
        :param image: Input image (numpy array).
        :param kernel_size: Size of the square kernel.
        :return: Filtered image.
        """
        pad_size = kernel_size // 2
        padded_image = np.pad(image, pad_size, mode='constant')
        
        filtered_image = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered_image[i, j] = np.mean(padded_image[i:i+kernel_size, j:j+kernel_size])
        
        return filtered_image.astype(np.uint8)

    def apply_gaussian_filter(self, image, kernel_size = 3, sigma = 10):
        """
        Apply Gaussian filter to the image.
        :param image: Input image (numpy array).
        :param kernel_size: Size of the square kernel.
        :param sigma: Standard deviation of the Gaussian distribution.
        :return: Filtered image.
        """        
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x)**2 + (y)**2) / (2*sigma**2)), (kernel_size, kernel_size))
        
        kernel /= np.sum(kernel)

        filtered_image = self.convolve(image, kernel)      
        
        return filtered_image

    def apply_median_filter(self, image, kernel_size=3):
        """
        Apply median filter to the image.
        :param image: Input image (numpy array).
        :param kernel_size: Size of the square kernel.
        :return: Filtered image.
        """
        pad_size = kernel_size // 2
        padded_image = np.pad(image, pad_size, mode='constant')
        
        filtered_image = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered_image[i, j] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size])
        
        return filtered_image.astype(np.uint8)
    





    
    def convolve(self, image, kernel):
        x, y = image.shape
        k = kernel.shape[0]
        convolved_image = np.zeros(shape=(x-2*k, y-2*k))
        for i in range(x-2*k):
            for j in range(y-2*k):
                mat = image[i:i+k, j:j+k]
                convolved_image[i, j] = np.sum(np.multiply(mat, kernel))

        return convolved_image


    def laplacian_edge(self, image, direction='both'):
        """
        Apply Laplacian edge detection to the image.
        :param image: Input image (numpy array).
        :param direction: Direction of edge detection ('x', 'y', or 'both').
        :return: Edge-detected image.
        """
        laplacian_kernel = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])
        return self.apply_edge(image, laplacian_kernel, direction)


    def prewitt_edge(self, image, direction='both'):
        """
        Apply Prewitt edge detection to the image.
        :param image: Input image (numpy array).
        :param direction: Direction of edge detection ('x', 'y', or 'both').
        :return: Edge-detected image.
        """
        prewitt_kernel = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])
        return self.apply_edge(image, prewitt_kernel, direction)


    def sobel_edge(self, image, direction='both'):
        """
        Apply Sobel edge detection to the image.
        :param image: Input image (numpy array).
        :param direction: Direction of edge detection ('x', 'y', or 'both').
        :return: Edge-detected image.
        """
        sobel_kernel = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])
        return self.apply_edge(image, sobel_kernel, direction)

    def roberts_edge(self, image, direction='both'):
        """
        Apply Roberts edge detection to the image.
        :param image: Input image (numpy array).
        :param direction: Direction of edge detection ('x', 'y', or 'both').
        :return: Edge-detected image.
        """
        roberts_kernel = np.array([[1, 0],
                                    [0, -1]])            
        return self.apply_edge(image, roberts_kernel, direction)
    

    def apply_edge(self, image, array, direction):
        if direction == 'Horizontal':
            edge = abs(self.convolve(image, array))
        elif direction == 'Vertical':
            edge = abs(self.convolve(image, array.T))
        else:
            edge_x = abs(self.convolve(image, array))
            edge_y = abs(self.convolve(image, array.T))
            edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge.astype(np.uint8)


    def convert_to_grayscale(self, image):
        """
        Convert the RGB image to grayscale using NTSC formula.
        :param image: Input RGB image (numpy array).
        :return: Grayscale image (numpy array).
        """
        rgb_coefficients = [0.299, 0.587, 0.114]
        grayscale_image = np.dot(image[..., :3], rgb_coefficients)

        return grayscale_image.astype(np.uint8)


    def get_RGB_histograms_and_cdf(self, image):
        """
        Compute the RGB histograms and cumulative distribution functions (CDFs) of the image.
        :param image: Input image (numpy array)
        :return: Tuple containing RGB histograms and CDFs.
        """
        if len(image.shape) == 2:
            hist = [0] * 256
            cdf = [0] * 256
            total_pixels = image.shape[0] * image.shape[1]

            for row in image:
                for pixel in row:
                    hist[pixel] += 1

            cdf[0] = hist[0] / total_pixels
            for i in range(1, 256):
                cdf[i] = cdf[i-1] + hist[i] / total_pixels

            return [hist, hist, hist], [cdf, cdf, cdf]

        elif len(image.shape) == 3 and image.shape[2] == 3:
            hist = [[0]*256, [0]*256, [0]*256]
            cdf = [[0]*256, [0]*256, [0]*256]
            total_pixels = image.shape[0] * image.shape[1]

            for row in image:
                for pixel in row:
                    for i in range(3):
                        hist[i][pixel[i]] += 1

            for i in range(3):
                cdf[i][0] = hist[i][0] / total_pixels
                for j in range(1, 256):
                    cdf[i][j] = cdf[i][j-1] + hist[i][j] / total_pixels

            return hist, cdf
        else:
            raise ValueError("Unsupported image format")
    # ==========================================================================================================================================
    # ================================================= Canny Edge Detector ====================================================================
    # ==========================================================================================================================================
        
    def canny_edge(self, image, direction='both', low_threshold=50, high_threshold=150):
        # Step 1: Apply Gaussian blur
        blurred_image = self.apply_gaussian_filter(image)

        # Step 2: Compute gradient intensity and direction
        gradient_magnitude, gradient_direction = self.compute_gradient(blurred_image)

        # Step 3: Non-maximum suppression
        suppressed_image = self.non_maximum_suppression(gradient_magnitude, gradient_direction)

        # Step 4: Double thresholding
        thresholded_image = self.double_thresholding(suppressed_image, low_threshold, high_threshold)

        # Step 5: Edge tracking by hysteresis
        canny_edges = self.edge_tracking(thresholded_image, low_threshold, high_threshold)

        return canny_edges


    def compute_gradient(self, image):
        sobel_x = self.convolve(image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        sobel_y = self.convolve(image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)

        return gradient_magnitude, gradient_direction


    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        suppressed_image = np.zeros_like(gradient_magnitude)

        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                angle = gradient_direction[i, j]
                q, r = 255, 255

                if (0 <= angle < np.pi / 8) or (7 * np.pi / 8 <= angle <= np.pi):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                elif (np.pi / 8 <= angle < 3 * np.pi / 8) or (5 * np.pi / 8 <= angle < 7 * np.pi / 8):
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8) or (5 * np.pi / 8 <= angle < 7 * np.pi / 8):
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]

                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    suppressed_image[i, j] = gradient_magnitude[i, j]

        return suppressed_image


    def double_thresholding(self, gradient_magnitude, low_threshold, high_threshold):
        strong_edges = gradient_magnitude > high_threshold
        weak_edges = (gradient_magnitude >= low_threshold) & (gradient_magnitude <= high_threshold)
        thresholded_image = np.zeros_like(gradient_magnitude)
        thresholded_image[strong_edges] = 255
        thresholded_image[weak_edges] = 50
        return thresholded_image


    def edge_tracking(self, thresholded_image, low_threshold, high_threshold):
        strong_edges = (thresholded_image == 255)
        weak_edges = (thresholded_image == 50)

        # Perform depth-first search to track weak edges
        def dfs(i, j):
            if 0 <= i < thresholded_image.shape[0] and 0 <= j < thresholded_image.shape[1]:
                if thresholded_image[i, j] == 50:
                    thresholded_image[i, j] = 255
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if (di != 0 or dj != 0):
                                dfs(i + di, j + dj)

        for i in range(thresholded_image.shape[0]):
            for j in range(thresholded_image.shape[1]):
                if strong_edges[i, j]:
                    dfs(i, j)

        # Set all remaining weak edges to zero
        thresholded_image[weak_edges] = 0

        return thresholded_image


    # ==========================================================================================================================================
    # ================================================== Hough Transforms ======================================================================
    # ==========================================================================================================================================

    def line_hough_transform(self):
        image = self.canny_edge(self.image)
        w, h = image.shape
        xy_pairs = np.nonzero(image)

        R_max = ceil(((w ** 2) + (h ** 2)) ** (0.5))

        thetas = np.deg2rad(np.arange(-90.0, 90.0))
        rhos = np.linspace(-R_max, R_max, R_max * 2)

        cosines, sines = np.cos(thetas), np.sin(thetas)

        accumulator = np.zeros((2 * R_max, len(thetas)), dtype=np.uint32)

        for (y, x) in zip(*xy_pairs):
            for theta in range(len(thetas)):
                rho = int(round(x * cosines[theta] + y * sines[theta]) + R_max)
                accumulator[rho, theta] += 1

        return accumulator, rhos, thetas
    

    def line_hough_peaks_suppresion(self, accumlator, num_peaks, nhood_size=3):
        indices = []
        hough_space = np.copy(accumlator)

        for _ in range(num_peaks):
            idx = np.argmax(hough_space)
            hough_space_idx = np.unravel_index(idx, hough_space.shape)
            indices.append(hough_space_idx)

            idx_y, idx_x = hough_space_idx
            min_x = max(0, idx_x - (nhood_size // 2))
            max_x = min(accumlator.shape[1], idx_x + (nhood_size // 2) + 1)
            min_y = max(0, idx_y - (nhood_size // 2))
            max_y = min(accumlator.shape[0], idx_y + (nhood_size // 2) + 1)

            hough_space[min_y:max_y, min_x:max_x] = 0

        return indices, hough_space


    def draw_lines(self, image, indices, rhos, thetas):
        image = np.stack((image,) * 3, axis=-1)
        
        for (idx_rhos, idx_thetas) in indices:
            rho = rhos[idx_rhos]
            theta = thetas[idx_thetas]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho

            offset = len(rhos) # Line Length
            x1, y1 = int(x0 + offset * (-b)), int(y0 + offset * (a))
            x2, y2 = int(x0 - offset * (-b)), int(y0 - offset * (a))

            line(image, (x1, y1), (x2, y2), (166,16,30), 2)

        return image
    

    def circle_hough_transform(self, min_radius, max_radius):
        image = self.canny_edge(self.image)
        w, h = image.shape
        xy_pairs = np.nonzero(image)

        thetas = np.deg2rad(np.arange(360))
        sines, cosines = np.sin(thetas), np.cos(thetas)
        hough_space = np.zeros((w, h, max_radius - min_radius + 1), dtype=np.uint8)

        for (y, x) in zip(*xy_pairs):
            for r in range(min_radius, max_radius + 1):
                a = y - np.round(r * cosines).astype(int)
                b = x - np.round(r * sines).astype(int)
                valid = (a >= 0) & (a < w) & (b >= 0) & (b < h)
                hough_space[a[valid], b[valid], r - min_radius] += 1

        return hough_space


    def circle_hough_peaks_suppression(self, accumulator, num_peaks, min_radius, nhood_size=3):
        peaks = []
        w, h, _ = accumulator.shape

        flat_accumulator = accumulator.reshape(-1)
        indices = np.argpartition(flat_accumulator, -num_peaks)[-num_peaks:]
        peak_coords = np.unravel_index(indices, accumulator.shape)

        for row, col, radius_offset in zip(*peak_coords):
            center_y, center_x = row, col
            radius = min_radius + radius_offset

            min_x = max(0, center_x - (nhood_size // 2))
            max_x = min(h, center_x + (nhood_size // 2) + 1)
            min_y = max(0, center_y - (nhood_size // 2))
            max_y = min(w, center_y + (nhood_size // 2) + 1)
            neighborhood = accumulator[min_y:max_y, min_x:max_x, :]

            if np.all(neighborhood <= accumulator[row, col, radius_offset]):
                peaks.append((row, col, radius))

        return peaks
    

    def draw_circles(self, circles):
        image = np.stack((self.image,) * 3, axis=-1)

        for row, col, radius in circles:
            circle(image, (col, row), radius, (166,16,30), 2)

        return image

    def ellipse_hough_transform(self, min_a, max_a, min_b, max_b, theta_resolution=1):
        edges = self.canny_edge(self.image)
        h, w = edges.shape

        accumulator = np.zeros((h, w, max_a - min_a + 1, max_b - min_b + 1), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                if not edges[y, x]:
                    continue

                # Loop through semi-major and semi-minor axis lengths
                for a in range(min_a, max_a + 1):
                    for b in range(min_b, max_b + 1):
                        # Generate range of orientations
                        thetas = np.arange(0, 360, theta_resolution, dtype=np.float32) * np.pi / 180

                        # Loop through orientations and vote in accumulator
                        for theta in thetas:
                            p = int(x + a * np.cos(theta))
                            q = int(y + b * np.sin(theta))

                            if 0 <= p < w and 0 <= q < h:
                                a_offset = a - min_a
                                b_offset = b - min_b
                                accumulator[q, p, a_offset, b_offset] += 1

        return accumulator


    def ellipse_hough_peaks_suppression(self, accumulator, num_peaks, min_a, min_b, threshold=None):
        peaks = []
        flat_accumulator = accumulator.reshape(-1)

        if threshold is None:
            indices = np.argpartition(flat_accumulator, -num_peaks)[-num_peaks:]
        else:
            indices = np.where(flat_accumulator >= threshold)[0]
            indices = indices[np.argsort(flat_accumulator[indices])[-num_peaks:]]

        peak_coords = np.unravel_index(indices, accumulator.shape)

        # Extract ellipse parameters from peaks
        for row, col, a_offset, b_offset in zip(*peak_coords):
            center_y, center_x = row, col
            semi_major_axis = min_a + a_offset
            semi_minor_axis = min_b + b_offset
            peaks.append((center_y, center_x, semi_major_axis, semi_minor_axis))

        return peaks
    
    def draw_ellipses(self, ellipses, color=(0, 255, 0), thickness=2):
        image = np.stack((self.image,) * 3, axis=-1)

        for center_y, center_x, semi_major_axis, semi_minor_axis in ellipses:
            cv2.ellipse(image, (int(center_x), int(center_y)), (int(semi_major_axis), int(semi_minor_axis)),
                        0, 0, 360, color, thickness)

        return image