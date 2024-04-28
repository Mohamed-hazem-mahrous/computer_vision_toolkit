import cv2
import numpy as np

class ImageMatching:
    def __init__(self, original_image, template_image):
        self.original_image_gray = original_image
        self.template_image_gray = template_image
        # self.method = method
        # self.original_image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # self.template_image_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
        self.height1, self.width1 = self.original_image_gray.shape
        self.height2, self.width2 = self.template_image_gray.shape
        # self.result = self.match_images()

    def compute_ssd(self, descriptor1, descriptor2):
        ssd = -(np.sqrt(np.sum((descriptor1 - descriptor2)**2)))
        return ssd

    def compute_ncc(self, descriptor1, descriptor2):
        normalized_original = (descriptor1 - np.mean(descriptor1)) / np.std(descriptor1)
        normalized_template = (descriptor2 - np.mean(descriptor2)) / np.std(descriptor2)
        ncc = float(np.mean(normalized_original * normalized_template))
        return ncc
    
    def downsample_images(self, scale_factor=0.5):
        self.original_image_gray = cv2.resize(self.original_image_gray, None, fx=scale_factor, fy=scale_factor)
        self.template_image_gray = cv2.resize(self.template_image_gray, None, fx=scale_factor, fy=scale_factor)
        
    def match_images(self, descriptor1, descriptor2, method='ssd'):
        num_keypoints1 = descriptor1.shape[0]
        num_keypoints2 = descriptor2.shape[0]

        matches_list = []

        for original_keypoint in range(num_keypoints1):
            best_match_keypoint = None
            best_score = -np.inf
            for template_keypoint in range(num_keypoints2):
                if method == 'ssd':
                    score = self.compute_ssd(descriptor1[original_keypoint], descriptor2[template_keypoint])
                elif method == 'ncc':
                    score = self.compute_ncc(descriptor1[original_keypoint], descriptor2[template_keypoint])
                else:
                    score = -np.inf

                if score > best_score:
                    best_score = score
                    best_match_keypoint = template_keypoint

            matched_feature = cv2.DMatch()

            matched_feature.queryIdx = original_keypoint
            matched_feature.trainIdx = best_match_keypoint
            matched_feature.distance = best_score

            matches_list.append(matched_feature)

        return matches_list

