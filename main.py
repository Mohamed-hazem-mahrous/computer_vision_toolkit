
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QSlider,QHBoxLayout , QLabel,QFileDialog
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
import os
import sys
import bisect
import pyqtgraph as pg
import numpy as np
import image_processing 
from PyQt5.QtGui import QPixmap,QImage
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, filePath):
        self.filePath = filePath
        self.image = cv2.imread(self.filePath, cv2.IMREAD_GRAYSCALE)
      
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
class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        self.image=None 
        super(MainWindow, self).__init__(*args, **kwargs)
        # Load the UI Page
        uic.loadUi(r'task1.ui', self)
        # type your code here 
        self.global_thresholding_slider.setMinimum(0)
        self.global_thresholding_slider.setMaximum(255)
        self.local_thresholding_slider.setMinimum(0)
        self.local_thresholding_slider.setMaximum(255)
        self.local_block_size_slider.setMinimum(1)
        self.browse_btn.clicked.connect(self.browseImage)
        self.global_thresholding_slider.valueChanged.connect(self.global_threshold_slider_value_changed)
        self.local_thresholding_slider.valueChanged.connect(self.local_threshold_sliders_value_changed)
        self.local_block_size_slider.valueChanged.connect(self.local_threshold_sliders_value_changed)

    def local_threshold_sliders_value_changed(self):
        block_size=self.local_block_size_slider.value()
        local_thresholding_val=self.local_thresholding_slider.value()
        self.display_image_in_label(self.local_image_label_page3,self.image.local_thresholding( block_size, local_thresholding_val) ) #display local thresholding image

    def global_threshold_slider_value_changed(self):
        global_thresholding_val=self.global_thresholding_slider.value()
        self.display_image_in_label(self.global_image_label_page3,self.image.global_thresholding(global_thresholding_val) ) #display global thresholding image

    def display_image_in_label(self,label, image):
        height, width = image.shape 
        channel = 1  # Set the channel to 1 for grayscale
        bytes_per_line = channel * width
        # Convert the image to QImage
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)
        # Resize the QPixmap to fit the QLabel while maintaining aspect ratio
        pixmap = pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio)
        # Set the pixmap to the label
        label.setPixmap(pixmap)
        
    def display_images_page3(self,image):
        self.display_image_in_label(self.original_image_label_page3,image.image ) #display original image
        self.display_image_in_label(self.normalized_image_label_page3,image.image_normalization() ) #display normalized image
        global_thresholding_val=self.global_thresholding_slider.value()
        self.display_image_in_label(self.global_image_label_page3,image.global_thresholding(global_thresholding_val) ) #display global thresholding image
        block_size=self.local_block_size_slider.value()
        local_thresholding_val=self.local_thresholding_slider.value()
        self.display_image_in_label(self.local_image_label_page3,image.local_thresholding( block_size, local_thresholding_val) ) #display local thresholding image

    def browseImage(self):
        # Open file dialog to select an image
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        self.image = ImageProcessor(filePath)
        self.display_images_page3(self.image)
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()