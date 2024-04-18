from functools import cmp_to_key
import numpy as np
import time
import cv2

class SIFT:
    def __init__(self, original_image):
        self.start_time = time.time()
        self.original_image = original_image.astype('float32')
        
        # Scale Space Parameters
        self.sigma = 1.6
        self.no_of_levels = 3
        self.assumed_blur = 0.5
        self.image_border_width = 5

        # Orientation Calculation Parameters
        self.radius_factor = 3
        self.peak_ratio = 0.8
        self.scale_factor = 1.5

    def sift(self):
        
        # Creating the Scale Space
        image = cv2.resize(self.original_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        sigma_diff = np.sqrt(max((self.sigma ** 2) - ((2 * self.assumed_blur) ** 2), 0.01))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
        num_octaves = int(np.round(np.log(min(image.shape)) / np.log(2) - 1))
        gaussian_pyramid = self.create_gaussian_pyramid(image, num_octaves)
        DoG_pyramid = self.create_DoG_pyramid(gaussian_pyramid)

        # Extracting Keypoints and Descriptors
        keypoints = self.localize_keypoints(gaussian_pyramid, DoG_pyramid)
        keypoints = self.remove_duplicate_keypoints(keypoints)
        keypoints = self.broadcast_keypoints_to_image_size(keypoints)

        print(f"SIFT Computation time: {time.time() - self.start_time}")


    def calculate_sigma_values(self):
        num_images_per_octave = self.no_of_levels + 3
        k = 2 ** (1. / self.no_of_levels)
        gaussian_kernels = np.zeros(num_images_per_octave)
        gaussian_kernels[0] = self.sigma

        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * self.sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        
        return gaussian_kernels


    def create_gaussian_pyramid(self, image, num_octaves):
        
        gaussian_pyramid = []
        sigmas = self.calculate_sigma_values()
        
        gaussian_pyramid = []
        for octave in range(num_octaves):
            gaussian_levels = []
            
            for sigma in sigmas:
                if len(gaussian_levels) == 0:
                    gaussian_levels.append(image)
                else:
                    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    gaussian_levels.append(image)
            
            gaussian_pyramid.append(gaussian_levels)

            if octave < num_octaves - 1:
                image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        
        return np.array(gaussian_pyramid, dtype=object)


    def create_DoG_pyramid(self, gaussian_pyramid):
        
        DoG_pyramid = []

        for gaussian_levels in gaussian_pyramid:
            DoG_octave = []

            for first_image, second_image in zip(gaussian_levels, gaussian_levels[1:]):
                DoG_octave.append(np.subtract(second_image, first_image))
            
            DoG_pyramid.append(DoG_octave)

        return np.array(DoG_pyramid, dtype=object)


    def localize_keypoints(self, gaussian_pyramid, DoG_pyramid, contrast_threshold=0.04):
        threshold = int(0.5 * contrast_threshold / self.no_of_levels * 255)
        keypoints = []

        for octave_index, DoG_octave in enumerate(DoG_pyramid):
            for image_index in range(1, len(DoG_octave) - 1):
                current_image, next_image, following_image = DoG_octave[image_index - 1: image_index + 2]
                for i in range(self.image_border_width, current_image.shape[0] - self.image_border_width):
                    for j in range(self.image_border_width, current_image.shape[1] - self.image_border_width):
                        if self.is_extreme(current_image[i - 1:i + 2, j - 1:j + 2],
                                            next_image[i - 1:i + 2, j - 1:j + 2],
                                            following_image[i - 1:i + 2, j - 1:j + 2],
                                            threshold):
                            localization_result = self.apply_quadratic_fit(i, j, image_index, octave_index, DoG_octave, contrast_threshold)
                            if localization_result is not None:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = self.get_keypoints_with_orientations(keypoint, octave_index, gaussian_pyramid[octave_index][localized_image_index])
                                keypoints.extend(keypoints_with_orientations)

        return keypoints


    def is_extreme(self, first_subimage, second_subimage, third_subimage, threshold):
        center_pixel_value = second_subimage[1, 1]
    
        if abs(center_pixel_value) <= threshold:
            return False

        if center_pixel_value > 0:
            return (
                center_pixel_value >= first_subimage.max() and
                center_pixel_value >= third_subimage.max() and
                center_pixel_value >= second_subimage.max()
            )
        else:
            return (
                center_pixel_value <= first_subimage.min() and
                center_pixel_value <= third_subimage.min() and
                center_pixel_value <= second_subimage.min()
            )


    def apply_quadratic_fit(self, i, j, image_index, octave_index, DoG_octave, contrast_threshold):
        
        for _ in range(5):
            first_image, second_image, third_image = DoG_octave[image_index-1:image_index+2]
            pixels_stack = np.stack([first_image[i-1:i+2, j-1:j+2],
                                     second_image[i-1:i+2, j-1:j+2],
                                     third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            
            extremum_update, gradient, hessian = self.calculate_gradient_hessian(pixels_stack)
            
            if np.all(np.abs(extremum_update) < 0.5):
                break

            i, j, image_index = self.update_indices(i, j, image_index, extremum_update)
        
            if self.is_outside_image(i, j, image_index, DoG_octave[0].shape):
                return None
        
        function_value = pixels_stack[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        
        if self.is_keypoint_valid(contrast_threshold, function_value, hessian):
            keypoint = self.initialize_keypoint(i, j, octave_index, image_index, extremum_update, function_value)
            return keypoint, image_index
            
        return None

    def calculate_gradient_hessian(self, pixels_stack):
        
        gradient = self.calculate_gradient_matrix(pixels_stack)
        hessian = self.calculate_hessian_matrix(pixels_stack)
        
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        return extremum_update, gradient, hessian

    def update_indices(self, i, j, image_index, extremum_update):
        j += int(np.round(extremum_update[0]))
        i += int(np.round(extremum_update[1]))
        image_index += int(np.round(extremum_update[2]))
        return i, j, image_index

    def is_outside_image(self, i, j, image_index, image_shape):
        return (i < self.image_border_width or i >= image_shape[0] - self.image_border_width or 
                j < self.image_border_width or j >= image_shape[1] - self.image_border_width or 
                image_index < 1 or image_index > self.no_of_levels)

    def is_keypoint_valid(self, contrast_threshold, function_value, hessian):
        eigenvalue_ratio = 12.1

        if abs(function_value) * self.no_of_levels < contrast_threshold:
            return False
        
        hessian = hessian[:2, :2]
        hessian_trace = np.trace(hessian)
        hessian_det = np.linalg.det(hessian)
        
        if hessian_det <= 0:
            return False
        
        return ((hessian_trace) ** 2 / hessian_det) < eigenvalue_ratio

    def initialize_keypoint(self, i, j, octave_index, image_index, extremum_update, function_value):
        keypoint = cv2.KeyPoint()
        keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
        keypoint.octave = octave_index + image_index * (2 ** 8) + int(np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
        keypoint.size = self.sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(self.no_of_levels))) * (2 ** (octave_index + 1))
        keypoint.response = abs(function_value)
        return keypoint

    def calculate_gradient_matrix(self, pixel_array):
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def calculate_hessian_matrix(self, pixel_array):
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs], 
                    [dxy, dyy, dys],
                    [dxs, dys, dss]])



    def get_keypoints_with_orientations(self, keypoint, octave_index, gaussian_image):
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        scale = self.scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
        radius = int(np.round(self.radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(36)
        smooth_histogram = np.zeros(36)

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                region_y = int(np.round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
                region_x = int(np.round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                
                if 0 < region_y < image_shape[0] - 1 and 0 < region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude, gradient_orientation = np.sqrt(dx * dx + dy * dy), np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    
                    histogram_index = int(np.round(gradient_orientation * 36 / 360.))
                    raw_histogram[histogram_index % 36] += weight * gradient_magnitude

        for n in range(36):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % 36]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % 36]) / 16.

        orientation_max = max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]

        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= self.peak_ratio * orientation_max:
                left_value = smooth_histogram[(peak_index - 1) % 36]
                right_value = smooth_histogram[(peak_index + 1) % 36]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % 36
                orientation = 360. - interpolated_peak_index * 360. / 36
                if abs(orientation - 360.) < 1e-7:
                    orientation = 0
                new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)

        return keypoints_with_orientations


    def compare_keypoints(self, keypoint1, keypoint2):
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    def remove_duplicate_keypoints(self, keypoints):
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self.compare_keypoints))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
            last_unique_keypoint.size != next_keypoint.size or \
            last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints



    def broadcast_keypoints_to_image_size(self, keypoints):
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints