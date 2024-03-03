
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
        self.global_thresholding_slider.sliderReleased.connect(self.global_threshold_slider_value_changed)
        self.local_thresholding_slider.sliderReleased.connect(self.local_threshold_sliders_value_changed)
        self.local_block_size_slider.sliderReleased.connect(self.local_threshold_sliders_value_changed)


        self.image_original_view = self.original_img_1
        self.image_original_view.setAspectLocked(True)
        self.image_original_view.setMouseEnabled(x=False, y=False)
        self.image_original_view.setMenuEnabled(False)
        self.image_original_view.hideAxis('left')
        self.image_original_view.hideAxis('bottom')
    
        self.image_original_view_edge = self.original_img_2
        self.image_original_view_edge.setAspectLocked(True)
        self.image_original_view_edge.setMouseEnabled(x=False, y=False)
        self.image_original_view_edge.setMenuEnabled(False)
        self.image_original_view_edge.hideAxis('left')
        self.image_original_view_edge.hideAxis('bottom')
                
        self.image_manipulated_view = self.manipulated_img_1
        self.image_manipulated_view.setAspectLocked(True)
        self.image_manipulated_view.setMouseEnabled(x=False, y=False)
        self.image_manipulated_view.setMenuEnabled(False)
        self.image_manipulated_view.hideAxis('left')
        self.image_manipulated_view.hideAxis('bottom')

        self.image_manipulated_view_edge = self.manipulated_img_2
        self.image_manipulated_view_edge.setAspectLocked(True)
        self.image_manipulated_view_edge.setMouseEnabled(x=False, y=False)
        self.image_manipulated_view_edge.setMenuEnabled(False)
        self.image_manipulated_view_edge.hideAxis('left')
        self.image_manipulated_view_edge.hideAxis('bottom')

        self.img_item_original = pg.ImageItem()
        self.img_item_manipulated = pg.ImageItem()
        self.image_original_view.addItem(self.img_item_original)
        self.image_manipulated_view.addItem(self.img_item_manipulated)

        self.img_item_original_edge = pg.ImageItem()
        self.img_item_manipulated_edge = pg.ImageItem()
        self.image_original_view_edge.addItem(self.img_item_original_edge)
        self.image_manipulated_view_edge.addItem(self.img_item_manipulated_edge)


        self.IMAGE = None
        self.img_path = None
        self.SNR = 0.01
        self.Kernel = 3
        self.last_operation = "Noise"


        self.noise_type_cb.currentIndexChanged.connect(self.apply_noise)

        self.NSR_Slider.sliderReleased.connect(self.change_SNR)

        self.Kernel_slider.sliderReleased.connect(self.change_Kernel)
        self.filter_type_cb.currentIndexChanged.connect(self.apply_filter)
    
        self.done_btn.clicked.connect(self.apply_edge_detection)

        




    def change_Kernel(self):
        self.Kernel = self.Kernel_slider.value()
        self.kernel_label.setText("Kernel Size: " + str(self.Kernel))
        self.apply_filter()
    
    def change_SNR(self):
        self.SNR = self.NSR_Slider.value() / 100
        self.SNR_label.setText(" Noise Ratio: " + str(self.SNR))
        self.apply_noise()

    def apply_edge_detection(self):
        edge_method_mapping = {
            "Sobel": "sobel_edge",
            "Prewitt": "prewitt_edge",
            "Roberts": "roberts_edge"
        }

        edge_filter = self.edge_filter_combobox.currentText()
        method_name = edge_method_mapping.get(edge_filter, "laplacian_edge")
        out = getattr(ImageProcessor(self.img_path), method_name)(
            image=self.IMAGE,
            direction=self.State_combobox.currentText()
        )
        self.img_item_manipulated_edge.setImage(out)
    
    def apply_filter(self):
        self.last_operation = "Filter"
        filter_method_mapping = {
            "Average": "apply_average_filter",
            "Median": "apply_median_filter",
            "Gaussian": "apply_gaussian_filter"
        }

        filter_type = self.filter_type_cb.currentText()
        method_name = filter_method_mapping.get(filter_type)
        if method_name:
            method = getattr(ImageProcessor(self.img_path), method_name)
            out = method(image=self.IMAGE, kernel_size=self.Kernel)
            self.img_item_manipulated.setImage(out)


    def apply_noise(self):
        self.last_operation = "Noise"
        noise_method_mapping = {
            "Uniform": ("add_uniform_noise", {"SNR": self.SNR}),
            "Gaussian": ("add_gaussian_noise", {"sigma": self.SNR}),
            "Salt and Pepper": ("add_salt_and_pepper_noise", {"amount": self.SNR})
        }

        noise_type = self.noise_type_cb.currentText()
        method_name, kwargs = noise_method_mapping.get(noise_type)
        if method_name:
            method = getattr(ImageProcessor(self.img_path), method_name)
            out = method(image=self.IMAGE, **kwargs)
            self.img_item_manipulated.setImage(out)

    
    def display_images_page1(self, img_path):
        self.IMAGE = cv2.rotate(self.convert_to_grayscale(cv2.imread(img_path)), cv2.ROTATE_90_CLOCKWISE)
        self.img_item_original.setImage(self.IMAGE)
        self.img_item_original_edge.setImage(self.IMAGE)

        if self.last_operation == "Noise":
            self.apply_noise()
        else:
            self.apply_filter()
    

    def convert_to_grayscale(self, image):
        rgb_coefficients = [0.299, 0.587, 0.114]
        grayscale_image = np.dot(image[..., :3], rgb_coefficients)

        return grayscale_image.astype(np.uint8)

    
    





    def browseImage(self):
        # Open file dialog to select an image
        self.img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        self.image = ImageProcessor(self.img_path)
        self.display_images_page3(self.image)
        self.display_images_page1(self.img_path)
        self.display_hist_dist(self.image)



        




    def local_threshold_sliders_value_changed(self):
        block_size=self.local_block_size_slider.value()
        local_thresholding_val=self.local_thresholding_slider.value()
        self.display_image_in_label(self.local_image_label_page3,self.image.local_thresholding( block_size, local_thresholding_val) ) #display local thresholding image

    def global_threshold_slider_value_changed(self):
        global_thresholding_val=self.global_thresholding_slider.value()
        self.display_image_in_label(self.global_image_label_page3,self.image.global_thresholding(global_thresholding_val) ) #display global thresholding image

    def display_image_in_label(self, label, image):
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
    

    
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()