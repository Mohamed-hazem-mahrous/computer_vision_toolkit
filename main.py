
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QSlider, QHBoxLayout, QLabel, QFileDialog, QVBoxLayout, QSizePolicy
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import sys
import bisect
import pyqtgraph as pg
from image_processing import ImageProcessor
from PyQt5.QtGui import QPixmap,QImage
import cv2
import numpy as np
from PIL import Image as PILImage



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

    def display_hist_dist(self, image):
        hist = image.get_histogram(image.image, 256)
        # cdf = image.get_cdf(hist, image.image.shape)
        self.display_histogram(hist)
        # self.display_cdf(cdf)

    def display_histogram(self, hist):
        self.histograme_plot.clear()
        self.histograme_plot.plot(hist, pen='r')

    def hybrid_images(self, image1, image2, alpha):
        image1 = PILImage.fromarray(image1)
        image2 = PILImage.fromarray(image2)

        if image1.size[0] * image1.size[1] > image2.size[0] * image2.size[1]:
            image1 = image1.resize((image2.size[0], image2.size[1]))
        else:
            image2 = image2.resize((image1.size[0], image1.size[1]))

        image1 = np.array(image1)
        image2 = np.array(image2)

        hybrid_image = (alpha * image1 + (1 - alpha) * image2).astype(np.uint8)

        return hybrid_image
    
    # def plot_histogram(self, hist):
    #     for i in range(len(hist)):
    #         self.histPlot.plot(hist[i], pen=(i, 3))
    

    def browseImage(self):
        # Open file dialog to select an image
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        self.image = ImageProcessor(filePath)
        self.display_images_page3(self.image)
        self.display_hist_dist(self.image)
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()