import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, uic
import sys
import pyqtgraph as pg
from image_processing import ImageProcessor
import numpy as np
from PIL import Image as PILImage
from Active_contour import Snake
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg 
import matplotlib.pyplot as plt


class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi(r'task1.ui', self)
        self.image_contour =()
        self.global_thresholding_slider.setMinimum(0)
        self.global_thresholding_slider.setMaximum(255)
        self.global_thresholding_slider.setValue(120)
        self.local_thresholding_slider.setMinimum(0)
        self.local_thresholding_slider.setMaximum(255)

        self.link_view_widgets()
        
        self.view_widgets = [self.manipulated_image_2, self.manipulated_image_1, self.original_image_2, self.original_image_1,
                            self.original_hybrid_image_1, self.original_hybrid_image_2, self.filtered_hybrid_image_1,
                            self.filtered_hybrid_image_2, self.filtered_hybrid_image_3, self.original_image_3, self.normalized_image,
                            self.local_thresholding_image, self.global_thresholding_image, self.original_image, self.equalized_image,
                            self.hough_transformed_image, self.original_image_5, self.manipulated_image_4, self.original_image_6]
        self.plot_widgets = [self.histograme_plot, self.distribution_curve_plot, self.R_Curve, self.G_Curve, self.B_Curve]
    
        for container in self.view_widgets:
            self.set_view_widget_settings(container)
    
        for container in self.plot_widgets:
            container.setBackground('w')
            container.setLimits(yMin = 0)


        for button in [self.browse_btn, self.upload_btn_1, self.upload_btn_2]:
            button.clicked.connect(lambda checked, btn=button: self.browse_image(btn))
        self.create_hybrid_btn.clicked.connect(self.hybrid_images)
        

        self.global_thresholding_slider.sliderReleased.connect(self.global_threshold_slider_value_changed)
        self.local_thresholding_slider.sliderReleased.connect(self.local_threshold_sliders_value_changed)
        self.local_block_size_slider.sliderReleased.connect(self.local_threshold_sliders_value_changed)
        self.NSR_Slider.valueChanged.connect(self.SNR_slider_value_changed)
        self.Kernel_slider.valueChanged.connect(self.kernel_slider_value_changed)
        self.NSR_Slider.sliderReleased.connect(self.apply_noise)
        self.Kernel_slider.sliderReleased.connect(self.apply_filter)
        self.hybrid_filter_slider_1.sliderReleased.connect(lambda: self.filter_radius_slider_value_changed(1))
        self.hybrid_filter_slider_2.sliderReleased.connect(lambda: self.filter_radius_slider_value_changed(2))
#_________________________________________________________________________________________________________
        self.Apply_btn.clicked.connect(  lambda : self.apply_contour())
        self.select_img_contour_btn.clicked.connect(lambda : self.display_images_page_contour())
        self.Reset_btn.clicked.connect(lambda: self.reset_contour())
#___________________________________________________________________________________________________________

        self.noise_type_cb.currentIndexChanged.connect(self.apply_noise)
        self.filter_type_cb.currentIndexChanged.connect(self.apply_filter)
        self.edge_filter_combobox.currentIndexChanged.connect(self.apply_edge_detection)
        self.State_combobox.currentIndexChanged.connect(self.apply_edge_detection)
        self.plotting_typr_combobox.currentIndexChanged.connect(self.display_hist_dist)
        self.filter_type_combobox_1.currentIndexChanged.connect(lambda: self.display_images_page6(1))
        self.filter_type_combobox_2.currentIndexChanged.connect(lambda: self.display_images_page6(2))

        # open_image_shortcut = QShortcut(Qt.CTRL + Qt.Key_O, self)
        # open_image_shortcut.activated.connect(self.browse_image)
        # self.setAcceptDrops(True)

        
        self.loaded_images = []
        
        self.SNR = 0.01
        self.Kernel = 3
        self.radius_lp_1, self.radius_hp_1 = 5, 10
        self.radius_lp_2, self.radius_hp_2 = 5, 10

        self.edge_method_mapping = {
            "Sobel": "sobel_edge",
            "Prewitt": "prewitt_edge",
            "Roberts": "roberts_edge",
            "Canny": "canny_edge"
        }
        self.filter_method_mapping = {
            "Average": "apply_average_filter",
            "Median": "apply_median_filter",
            "Gaussian": "apply_gaussian_filter"
        }
        
        self.label_texts = { "Uniform": "SNR", "Gaussian": "Sigma", "Salt and Pepper": "S & P amount" }

        self.line_hough_parameters = {
            'Hough Space': None,
            'Rhos': None,
            'Thetas': None
            }
        self.circle_hough_parameters = {
            'Hough Space': None,
            'min radius': 10,
            'max radius': 30
            }
        
        self.no_of_peaks_slider.valueChanged.connect(self.number_of_peaks_slider_value_changed)
        
        self.browse_btn_4.clicked.connect(self.browse_hough_image)
        self.hough_apply_btn.clicked.connect(self.apply_hough)

        self.line_hough_radio_btn.setChecked(True)

        for radio_btn in [self.line_hough_radio_btn, self.circle_hough_radio_btn, self.ellipse_hough_radio_btn]:
            radio_btn.clicked.connect(self.hough_radio_btn_clicked)

    
    def apply_line_hough_transform(self, hough_image, num_peaks=15, calculate_accumlator = True):
        image = hough_image
        if calculate_accumlator:
            (self.line_hough_parameters['Hough Space'], 
            self.line_hough_parameters['Rhos'], 
            self.line_hough_parameters['Thetas']) = image.line_hough_transform()
        
        indicies, _ = image.line_hough_peaks_suppresion(self.line_hough_parameters['Hough Space'], num_peaks, nhood_size=20)
        detected_image = image.draw_lines(image.image, indicies, self.line_hough_parameters['Rhos'], self.line_hough_parameters['Thetas'])
        
        self.hough_transform_view_widget.setImage(np.rot90(detected_image, k=-1))
        self.original_image_view_widget_hough.setImage(np.rot90(image.image, k=-1))


    def apply_circle_hough_transform(self, hough_image, num_peaks=15, calculate_accumlator = True):
        image = hough_image
        if calculate_accumlator:
            self.circle_hough_parameters['Hough Space'] = image.circle_hough_transform(self.circle_hough_parameters['min radius'], self.circle_hough_parameters['max radius'])
            
        circles = image.circle_hough_peaks_suppression(self.circle_hough_parameters['Hough Space'], num_peaks, self.circle_hough_parameters['min radius'])
        detected_image = image.draw_circles(circles)
        
        self.hough_transform_view_widget.setImage(np.rot90(detected_image, k=-1))
        self.original_image_view_widget_hough.setImage(np.rot90(image.image, k=-1))


    def number_of_peaks_slider_value_changed(self):
        value = self.no_of_peaks_slider.value()
        self.no_of_peaks_label.setText(f"No. of Peaks: {str(value)}")
        
        if self.line_hough_radio_btn.isChecked():
            calculate_accumlator_flag = self.line_hough_parameters['Hough Space'] is None
            self.apply_line_hough_transform(self.original_hough_image, num_peaks=value, calculate_accumlator=calculate_accumlator_flag)

        elif self.circle_hough_radio_btn.isChecked():
            calculate_accumlator_flag = self.circle_hough_parameters['Hough Space'] is None
            self.apply_circle_hough_transform(self.original_hough_image, num_peaks=value, calculate_accumlator=calculate_accumlator_flag)

    
    def browse_hough_image(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        initial_folder = os.path.join(script_directory, "Images")
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", initial_folder, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        self.original_hough_image = ImageProcessor(path)

        if self.line_hough_radio_btn.isChecked():
            self.apply_line_hough_transform(self.original_hough_image)

        elif self.circle_hough_radio_btn.isChecked():
            self.apply_circle_hough_transform(self.original_hough_image)

    def hough_radio_btn_clicked(self):
        self.number_of_peaks_slider_value_changed()

    def apply_hough(self):
        if self.line_hough_radio_btn.isChecked():
            return
        
        elif self.circle_hough_radio_btn.isChecked():
            self.circle_hough_parameters['min radius'] = int(self.min_radius_line_edit.text())
            self.circle_hough_parameters['max radius'] = int(self.max_radius_line_edit.text())
            self.apply_circle_hough_transform(self.original_hough_image, num_peaks=self.no_of_peaks_slider.value(), calculate_accumlator=True)


    def browse_image(self, button):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        initial_folder = os.path.join(script_directory, "Images")
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", initial_folder, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        self.load_image(path, button)
        
    def load_image(self, path, button):
        if button in [self.browse_btn, self.upload_btn_1]:
            if len(self.loaded_images) != 0:
                del self.loaded_images[0:]
            self.loaded_images.append(ImageProcessor(path))
            self.display_images_page1()
            self.display_images_page3()
            self.display_images_page6(1)
            self.display_hist_dist()
            self.apply_noise()
            self.apply_edge_detection()
            self.display_images_page4()
        else:
            self.loaded_images.append(ImageProcessor(path))
            del self.loaded_images[1:-1]
            self.display_images_page6(2)


    def select_image_for_contour(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        initial_folder = os.path.join(script_directory, "Images")
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", initial_folder, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        image = ImageProcessor(path).image
        self.image_contour =(image,path)
        return image

    def display_images_page_contour(self):
        self.reset_contour()
        self.output_image_contour.clear()
        image = self.select_image_for_contour()
        for view_widget in [self.original_image_contour, self.output_image_contour]:
            view_widget.setImage(np.rot90(image, k = -1))
        
    #_________________________________________________________________________
    def get_lineEdit_val(self):
        alpha = float(self.Alpha_lineEdit.text())
        beta  = float(self.Beta_lineEdit.text())
        gamma  = float(self.Gamma_lineEdit.text())
        itirations  = int(self.Iteration_lineEdit.text())
        return alpha,beta,gamma,itirations
    
    def plot_contour(self, cont_x, cont_y):
        # Extract contour coordinates
        contour_x = np.array(cont_x)
        contour_y = np.array(cont_y)

        # Clear existing contour if it exists
        if hasattr(self, 'contour_plot'):
            self.manipulated_image_4.removeItem(self.contour_plot)

        # Plot the new contour
        self.contour_plot = self.manipulated_image_4.plot(np.r_[cont_x, cont_x[0]], np.r_[cont_y, cont_y[0]], symbol='o', symbolSize=5, pen=pg.mkPen(color='r'))  # 'r' for red color
        #print("cont_x" , cont_x[1:5] )


    def apply_contour(self):
        alpha, beta, gamma, iterations = self.get_lineEdit_val()
        # Create a Snake instance
        snake_instance = Snake(self.image_contour[0], self.image_contour[1], alpha, beta, gamma, iterations)
        source = snake_instance.image
        contour_x = snake_instance.contour_x
        contour_y= snake_instance.contour_y
        external_energy = snake_instance.external_energy
        window_coordinates =snake_instance.window
        for i in range(iterations):
        # Start Applying Active Contour Algorithm
            cont_x, cont_y = snake_instance.update_contour(source, contour_x, contour_y,
                                            external_energy, window_coordinates)
            # snake_instance.contour_x=cont_x
            # snake_instance.contour_y =cont_y
            print(i)
            #print("cont_x_from_elly_ta7t" , cont_x[1:5] )
            self.plot_contour(cont_x,cont_y)

    def reset_contour(self):
        if hasattr(self, 'contour_plot'):
            self.manipulated_image_4.removeItem(self.contour_plot)
            # Clear the contents of line edits
        self.Alpha_lineEdit.clear()
        self.Beta_lineEdit.clear()
        self.Gamma_lineEdit.clear()
        self.Iteration_lineEdit.clear()
        self.image_contour= ()
       
    def kernel_slider_value_changed(self):
        self.Kernel = self.Kernel_slider.value()
        self.kernel_label.setText("Kernel Size: " + str(self.Kernel))


    def SNR_slider_value_changed(self):
        self.SNR = self.NSR_Slider.value() / 100
        snr_value_text = self.label_texts.get(self.noise_type_cb.currentText(), "")
        self.SNR_label.setText(f"{snr_value_text}: {str(self.SNR)}" )



    def apply_edge_detection(self):
        method_name = self.edge_method_mapping.get(self.edge_filter_combobox.currentText())
        out = getattr(self.loaded_images[0], method_name)(
            image=self.loaded_images[0].image,
            direction=self.State_combobox.currentText()
        )
        self.edge_manipulated_image_view_widget.setImage(np.rot90(out, k=-1))
    


    def apply_filter(self):
        method_name = self.filter_method_mapping.get(self.filter_type_cb.currentText())
        if method_name:
            method = getattr(self.loaded_images[0], method_name)
            out = method(image=self.loaded_images[0].noisy_image, kernel_size=self.Kernel)
            self.filter_manipulated_image_view_widget.setImage(np.rot90(out, k=-1))



    def apply_noise(self):
        self.noise_method_mapping = {
            "Uniform": ("add_uniform_noise", {"SNR": self.SNR}),
            "Gaussian": ("add_gaussian_noise", {"sigma": self.SNR}),
            "Salt and Pepper": ("add_salt_and_pepper_noise", {"amount": self.SNR})
        }

        snr_value_text = self.label_texts.get(self.noise_type_cb.currentText(), "")
        self.SNR_label.setText(f"{snr_value_text}: {str(self.SNR)}" )       

        method_name, kwargs = self.noise_method_mapping.get(self.noise_type_cb.currentText())
        if method_name:
            method = getattr(self.loaded_images[0], method_name)
            out = method(image=self.loaded_images[0].image, **kwargs)
            self.original_image_view_widget.setImage(np.rot90(out, k=-1))

        self.apply_filter()
    


    def display_images_page1(self):
        for view_widget in [self.original_image_view_widget, self.original_image_view_widget_edge]:
            view_widget.setImage(np.rot90(self.loaded_images[0].image, k = -1))



    def local_threshold_sliders_value_changed(self):
        block_size=self.local_block_size_slider.value()
        local_thresholding_val=self.local_thresholding_slider.value()
        # print(local_thresholding_val)
        # print(block_size)
        self.local_threshold_image_view_widget.setImage(np.rot90(self.loaded_images[0].local_thresholding( block_size, local_thresholding_val), k=-1))



    def global_threshold_slider_value_changed(self):
        global_thresholding_val=self.global_thresholding_slider.value()
        self.global_threshold_image_view_widget.setImage(np.rot90(self.loaded_images[0].global_thresholding(global_thresholding_val), k=-1))



    def display_images_page3(self):
        self.local_block_size_slider.setMinimum(1)
        local_block_size_slider_max_value =max(self.loaded_images[0].image.shape[0], self.loaded_images[0].image.shape[1])
        self.local_block_size_slider.setMaximum(local_block_size_slider_max_value)
        self.local_block_size_slider.setValue(int(local_block_size_slider_max_value/2))
        
        self.original_image_view_widget_thresh.setImage(np.rot90(self.loaded_images[0].image, k=-1))
        self.normalized_image_view_widget.setImage(np.rot90(self.loaded_images[0].image_normalization(), k=-1))
        
        global_thresholding_val=self.global_thresholding_slider.value()
        
        self.global_threshold_image_view_widget.setImage(np.rot90(self.loaded_images[0].global_thresholding(global_thresholding_val), k=-1))
        
        block_size=self.local_block_size_slider.value()
        local_thresholding_val=self.local_thresholding_slider.value()

        self.local_threshold_image_view_widget.setImage(np.rot90(self.loaded_images[0].local_thresholding( block_size, local_thresholding_val), k=-1))


    def display_images_page4(self):
        self.original_image_view_widget_eq.setImage(np.rot90(self.loaded_images[0].image, k=-1))
        self.equalized_image_view_widget.setImage(
            np.rot90(self.loaded_images[0].histogram_equalization(self.loaded_images[0].image, np.amax(self.loaded_images[0].image.flatten())), k=-1))


    def display_hist_dist(self):
        hist = self.loaded_images[0].get_histogram(self.loaded_images[0].image, 256)
        self.display_histogram(hist)
        self.display_distribution_curve()

        # Page 5 Plots
        if self.plotting_typr_combobox.currentText() == "Histogram":
            histograms_cdf = [hist for hist in self.loaded_images[0].RGBhistograms]
        else:
            histograms_cdf = [hist for hist in self.loaded_images[0].RGBcdf]
        
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for plot_widget, histogram, color in zip([self.G_Curve, self.B_Curve, self.R_Curve], histograms_cdf, colors):
            plot_widget.clear()
            plot_widget.plot(histogram, pen=color, fillLevel=-0.3, fillBrush=color + (80,))



    def display_histogram(self, hist):
        self.histograme_plot.clear()
        self.histograme_plot.plot(hist, pen='r')
        self.histograme_plot.setLabel('left', 'Frequency')
        self.histograme_plot.setLabel('bottom', 'Pixel Intensity')


    def display_distribution_curve(self):
        cdf = self.loaded_images[0].get_cdf(self.loaded_images[0].get_histogram(self.loaded_images[0].image.flatten(), 256), self.loaded_images[0].image.shape)
        self.distribution_curve_plot.clear()
        self.distribution_curve_plot.plot(cdf, pen='r')
        self.distribution_curve_plot.setLabel('left', 'Probability')
        self.distribution_curve_plot.setLabel('bottom', 'Pixel Intensity')


    def hybrid_images(self):
        alpha = 0.5
        image1 = PILImage.fromarray(self.image_to_be_mixed_1)
        image2 = PILImage.fromarray(self.image_to_be_mixed_2)

        if image1.size[0] * image1.size[1] > image2.size[0] * image2.size[1]:
            image1 = image1.resize((image2.size[0], image2.size[1]))
        else:
            image2 = image2.resize((image1.size[0], image1.size[1]))

        image1 = np.array(image1)
        image2 = np.array(image2)

        hybrid_image = (alpha * image1 + (1 - alpha) * image2).astype(np.uint8)
        self.hybrid_result_image.setImage(np.rot90(hybrid_image, k=-1))



    def display_images_page6(self, target):
        if self.loaded_images:
            target_image = self.loaded_images[0] if target == 1 else self.loaded_images[-1]
            filtered_lp, filtered_hp = self.get_frequency_domain_filters(target_image, getattr(self, f"radius_lp_{target}"), getattr(self, f"radius_hp_{target}"))

            filters_map = {"Low-pass filter": filtered_lp, "High-pass filter": filtered_hp}

            if target == 1:
                self.image_to_be_mixed_1 = filters_map[self.filter_type_combobox_1.currentText()]
                self.hybrid_image_1_filtered.setImage(np.rot90(self.image_to_be_mixed_1, k=-1))
                self.hybrid_image_1.setImage(np.rot90(target_image.image, k=-1))
            else:
                self.image_to_be_mixed_2 = filters_map[self.filter_type_combobox_2.currentText()]
                self.hybrid_image_2.setImage(np.rot90(target_image.image, k=-1))
                self.hybrid_image_2_filtered.setImage(np.rot90(self.image_to_be_mixed_2, k=-1))



    def filter_radius_slider_value_changed(self, target):
        filter_combobox = self.filter_type_combobox_1 if target == 1 else self.filter_type_combobox_2
        filter_type = filter_combobox.currentText()

        radius_attributes = {
            "Low-pass filter": f"radius_lp_{target}",
            "High-pass filter": f"radius_hp_{target}"
        }

        radius_attribute = radius_attributes.get(filter_type)
        setattr(self, radius_attribute, getattr(self, f"hybrid_filter_slider_{target}").value())

        self.display_images_page6(target)



    def get_frequency_domain_filters(self, target_image, radius_lp, radius_hp):
        f_transform = np.fft.fft2(target_image.image)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = target_image.image.shape
        crow, ccol = rows // 2, cols // 2
        lp_mask = np.zeros((rows, cols), np.uint8)
        lp_mask[crow - radius_lp:crow + radius_lp, ccol - radius_lp:ccol + radius_lp] = 1

        hp_mask = np.ones((rows, cols), np.uint8)
        hp_mask[crow - radius_hp:crow + radius_hp, ccol - radius_hp:ccol + radius_hp] = 0

        f_shift_lp = f_shift * lp_mask
        f_shift_hp = f_shift * hp_mask

        f_ishift_lp = np.fft.ifftshift(f_shift_lp)
        filtered_lp = np.fft.ifft2(f_ishift_lp)
        filtered_lp = np.abs(filtered_lp)

        f_ishift_hp = np.fft.ifftshift(f_shift_hp)
        filtered_hp = np.fft.ifft2(f_ishift_hp)
        filtered_hp = np.abs(filtered_hp)

        return filtered_lp, filtered_hp


    # utility function just to remove the bachground color of the pyqtgraph widgets and hide the axies
    def set_view_widget_settings(self, container):
        container.setBackground('#dddddd')
        container.setAspectLocked(True)
        container.hideAxis('left')
        container.hideAxis('bottom')

    # this is a function to create an image item and then links it with its viewer widget 
    def link_view_widgets(self):
        self.original_image_view_widget, self.filter_manipulated_image_view_widget = pg.ImageItem(), pg.ImageItem()
        self.original_image_1.addItem(self.original_image_view_widget)
        self.manipulated_image_1.addItem(self.filter_manipulated_image_view_widget)

        self.original_image_view_widget_edge, self.edge_manipulated_image_view_widget = pg.ImageItem(), pg.ImageItem()
        self.original_image_2.addItem(self.original_image_view_widget_edge)
        self.manipulated_image_2.addItem(self.edge_manipulated_image_view_widget)

        self.original_image_view_widget_thresh, self.normalized_image_view_widget = pg.ImageItem(), pg.ImageItem()
        self.original_image_3.addItem(self.original_image_view_widget_thresh)
        self.normalized_image.addItem(self.normalized_image_view_widget)

        self.local_threshold_image_view_widget, self.global_threshold_image_view_widget = pg.ImageItem(), pg.ImageItem()
        self.local_thresholding_image.addItem(self.local_threshold_image_view_widget)
        self.global_thresholding_image.addItem(self.global_threshold_image_view_widget)
        
        self.hybrid_image_1, self.hybrid_image_2 = pg.ImageItem(), pg.ImageItem()
        self.original_hybrid_image_1.addItem(self.hybrid_image_1)
        self.original_hybrid_image_2.addItem(self.hybrid_image_2)

        self.hybrid_image_1_filtered, self.hybrid_image_2_filtered = pg.ImageItem(), pg.ImageItem()
        self.hybrid_result_image = pg.ImageItem()
        self.filtered_hybrid_image_1.addItem(self.hybrid_image_1_filtered)
        self.filtered_hybrid_image_2.addItem(self.hybrid_image_2_filtered)
        self.filtered_hybrid_image_3.addItem(self.hybrid_result_image)

        self.original_image_view_widget_eq, self.equalized_image_view_widget = pg.ImageItem(), pg.ImageItem()
        self.original_image.addItem(self.original_image_view_widget_eq)
        self.equalized_image.addItem(self.equalized_image_view_widget)

        self.original_image_contour, self.output_image_contour = pg.ImageItem(), pg.ImageItem()
        self.original_image_5.addItem(self.original_image_contour)
        self.manipulated_image_4.addItem(self.output_image_contour)

        self.hough_transform_view_widget, self.original_image_view_widget_hough = pg.ImageItem(), pg.ImageItem()
        self.hough_transformed_image.addItem(self.hough_transform_view_widget)
        self.original_image_6.addItem(self.original_image_view_widget_hough)
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()