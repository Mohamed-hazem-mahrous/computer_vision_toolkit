import cv2
import numpy as np
from PIL import Image as PILImage

class SIFT:
    def __init__(self, original_image):
        self.original_image = original_image
        self.sift()

    def sift(self):
        first_octave_gaussian_levels, second_octave_gaussian_levels, third_octave_gaussian_levels, fourth_octave_gaussian_levels\
        = self.generate_scale_space(self.original_image)

        [first_octave_DoG_levels, second_DoG_gaussian_levels, third_DoG_gaussian_levels, fourth_DoG_gaussian_levels] =\
            self.calculate_DoG([first_octave_gaussian_levels, second_octave_gaussian_levels, third_octave_gaussian_levels, fourth_octave_gaussian_levels])
        
        cv2.imwrite("./Images/1stTry.png", fourth_DoG_gaussian_levels[-1])
        cv2.imwrite("./Images/2ndTry.png", first_octave_DoG_levels[1])


    def generate_scale_space(self, original_image):
        image = np.copy(original_image)

        first_octave = PILImage.fromarray(image)
        second_octave = first_octave.resize((first_octave.size[0] // 2, first_octave.size[1] // 2), PILImage.LANCZOS)
        third_octave = second_octave.resize((second_octave.width // 2, second_octave.height // 2), PILImage.LANCZOS)
        fourth_octave = third_octave.resize((third_octave.width // 2, third_octave.height // 2), PILImage.LANCZOS)

        # cv2.imwrite("./Images/1stTry.png", first_octave_gaussian_levels[-1])
        # cv2.imwrite("./Images/2ndTry.png", first_octave_gaussian_levels[1])

        return  self.generate_gaussian_levels([np.array(first_octave), np.array(third_octave), np.array(third_octave), np.array(fourth_octave)])



    def generate_gaussian_levels(self, octaves, kernel_size=15):
        gaussian_levels = []
        for image in octaves:
            tmp_gaussians = []
        
            for sigma in [1.0, 2.5, 5.1, 10.2, 25.6]:
                tmp_gaussians.append(self.apply_gaussian_filter(image, kernel_size=kernel_size, sigma=sigma))
            
            gaussian_levels.append(tmp_gaussians)
        
        return gaussian_levels
    

    def calculate_DoG(self, gaussian_levels):
        DoG_levels = []
        for gaussian_level in gaussian_levels:
            tmp_DoG = []
            for i in range(1, 5):
                tmp_DoG.append(gaussian_level[i] - gaussian_level[i - 1])
            
            DoG_levels.append(tmp_DoG)
        
        return DoG_levels
    

    def apply_gaussian_filter(self, image, kernel_size = 3, sigma = 10):
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x)**2 + (y)**2) / (2*sigma**2)), (kernel_size, kernel_size))
        kernel /= np.sum(kernel)
        filtered_image = self.convolve(image, kernel)      
        
        return filtered_image
    
    def convolve(self, image, kernel):
        x, y = image.shape
        k = kernel.shape[0]
        convolved_image = np.zeros(shape=(x-2*k, y-2*k))
        for i in range(x-2*k):
            for j in range(y-2*k):
                mat = image[i:i+k, j:j+k]
                convolved_image[i, j] = np.sum(np.multiply(mat, kernel))

        return convolved_image

class Keypoint:
      def __init__(self, x, y, scale, response, orientation):
        self.x = x
        self.y = y
        self.scale = scale
        self.response = response
        self.orientation = orientation
    

image = cv2.imread("./Images/CT head.jpg", cv2.IMREAD_GRAYSCALE)

SIFT(image)