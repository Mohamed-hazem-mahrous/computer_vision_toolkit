import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QShortcut
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets, uic
import sys
import pyqtgraph as pg
from image_processing import ImageProcessor
import numpy as np
from PIL import Image as PILImage


class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi(r'task1.ui', self)
        
        self.global_thresholding_slider.setMinimum(0)
        self.global_thresholding_slider.setMaximum(255)
        self.local_thresholding_slider.setMinimum(0)
        self.local_thresholding_slider.setMaximum(255)
        self.local_block_size_slider.setMinimum(1)

        self.link_view_widgets()
        
        self.view_widgets = [self.manipulated_image_2, self.manipulated_image_1, self.original_image_2, self.original_image_1,
                             self.original_hybrid_image_1, self.original_hybrid_image_2, self.filtered_hybrid_image_1,
                             self.filtered_hybrid_image_2, self.filtered_hybrid_image_3, self.original_image_3, self.normalized_image,
                             self.local_thresholding_image, self.global_thresholding_image]
        self.plot_widgets = [self.histograme_plot, self.distribution_curve_plot, self.R_Curve, self.G_Curve, self.B_Curve]
    
        for container in self.view_widgets:
            self.set_view_widget_settings(container)
    
        for container in self.plot_widgets:
            container.setBackground('w')
            container.setLimits(yMin = 0)


        for button in [self.browse_btn, self.upload_btn_1, self.upload_btn_2]:
            button.clicked.connect(self.browse_image)
        self.done_btn.clicked.connect(self.apply_edge_detection)
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


        self.noise_type_cb.currentIndexChanged.connect(self.apply_noise)
        self.filter_type_cb.currentIndexChanged.connect(self.apply_filter)
        self.edge_filter_combobox.currentIndexChanged.connect(self.apply_edge_detection)
        self.State_combobox.currentIndexChanged.connect(self.apply_edge_detection)
        self.plotting_typr_combobox.currentIndexChanged.connect(self.display_hist_dist)
        self.filter_type_combobox_1.currentIndexChanged.connect(lambda: self.display_images_page6(1))
        self.filter_type_combobox_2.currentIndexChanged.connect(lambda: self.display_images_page6(2))

        open_image_shortcut = QShortcut(Qt.CTRL + Qt.Key_O, self)
        open_image_shortcut.activated.connect(self.browse_image)

        
        self.loaded_images = []
        
        self.SNR = 0.01
        self.Kernel = 3
        self.radius_lp_1, self.radius_hp_1 = 5, 10
        self.radius_lp_2, self.radius_hp_2 = 5, 10

        self.edge_method_mapping = {
            "Sobel": "sobel_edge",
            "Prewitt": "prewitt_edge",
            "Roberts": "roberts_edge"
        }
        self.filter_method_mapping = {
            "Average": "apply_average_filter",
            "Median": "apply_median_filter",
            "Gaussian": "apply_gaussian_filter"
        }
        
        self.label_texts = { "Uniform": "SNR", "Gaussian": "Sigma", "Salt and Pepper": "S & P amount" }



    def browse_image(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        initial_folder = os.path.join(script_directory, "Images")
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", initial_folder, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        self.loaded_images.append(ImageProcessor(path))
        if len(self.loaded_images) == 1:
            self.display_images_page1()
            self.display_images_page3()
            self.display_images_page6(1)
            self.display_hist_dist()
            self.apply_noise()
            self.apply_edge_detection()
        else:
            del self.loaded_images[1:-1]
            self.display_images_page6(2)



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
        self.local_threshold_image_view_widget.setImage(np.rot90(self.loaded_images[0].local_thresholding( block_size, local_thresholding_val), k=-1))



    def global_threshold_slider_value_changed(self):
        global_thresholding_val=self.global_thresholding_slider.value()
        self.global_threshold_image_view_widget.setImage(np.rot90(self.loaded_images[0].global_thresholding(global_thresholding_val), k=-1))



    def display_images_page3(self):
        self.original_image_view_widget_thresh.setImage(np.rot90(self.loaded_images[0].image, k=-1))
        self.normalized_image_view_widget.setImage(np.rot90(self.loaded_images[0].image_normalization(), k=-1))
        
        global_thresholding_val=self.global_thresholding_slider.value()
        
        self.global_threshold_image_view_widget.setImage(np.rot90(self.loaded_images[0].global_thresholding(global_thresholding_val), k=-1))
        
        block_size=self.local_block_size_slider.value()
        local_thresholding_val=self.local_thresholding_slider.value()
        
        self.local_threshold_image_view_widget.setImage(np.rot90(self.loaded_images[0].local_thresholding( block_size, local_thresholding_val), k=-1))
        


    def display_hist_dist(self):
        hist = self.loaded_images[0].get_histogram(self.loaded_images[0].image, 256)
        # cdf = image.get_cdf(hist, image.image.shape)
        # self.display_cdf(cdf)
        self.display_histogram(hist)

        # Page 5 Plots
        if self.plotting_typr_combobox.currentText() == "Histogram":
            histograms_cdf = [hist for hist in self.loaded_images[0].RGBhistograms]
        else:
            histograms_cdf = [hist for hist in self.loaded_images[0].RGBcdf]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for plot_widget, histogram, color in zip([self.R_Curve, self.G_Curve, self.B_Curve], histograms_cdf, colors):
            plot_widget.clear()
            plot_widget.plot(histogram, pen=color, fillLevel=-0.3, fillBrush=color + (50,))



    def display_histogram(self, hist):
        self.histograme_plot.clear()
        self.histograme_plot.plot(hist, pen='r')



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



    def set_view_widget_settings(self, container):
        container.setBackground('#dddddd')
        container.setAspectLocked(True)
        # container.setMouseEnabled(x=False, y=False)
        # container.setMenuEnabled(False)
        container.hideAxis('left')
        container.hideAxis('bottom')



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

        
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()