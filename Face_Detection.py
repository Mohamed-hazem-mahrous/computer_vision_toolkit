from PyQt5.QtGui import QColor
import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, uic 
import sys
import pyqtgraph as pg
import numpy as np
import cv2
from Face_Recognition import pca, recognize_face, create_dataset
from PyQt5.QtCore import QCoreApplication, Qt, QDir

class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi(r'task5.ui', self)
        
        self.view_widgets = [self.image_detection_widget, self.recognition_image_widget]
        
        for container in self.view_widgets:
            self.set_view_widget_settings(container)

        self.detection_browse_button.clicked.connect(self.browse_image)
        self.recognition_browse_button.clicked.connect(self.browse_image)

        self.load_dataset_button.clicked.connect(self.browse_dataset)

        self.link_view_widgets()

        self.load_dataset()

        self.setAcceptDrops(True)

    def browse_image(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        initial_folder = os.path.join(script_directory, "Images")
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", initial_folder, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if path:
            if self.tabWidget.currentIndex() == 0:
                self.apply_face_detection(path)
            if self.tabWidget.currentIndex() == 1:
                QCoreApplication.processEvents()
                self.apply_face_recognition(path)

    def browse_dataset(self):
        initial_folder = QDir.currentPath()
        dataset_folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", initial_folder)
        if dataset_folder:
            print("Selected Dataset Folder:", dataset_folder)
            create_dataset(dataset_folder)
            self.load_dataset()

    def load_dataset(self):
        self.dataset = np.load('./dataset/images_dataset.npy')
        self.class_labels = np.load('./dataset/class_labels.npy')
        self.eigen_faces, self.centered_data, self.mean_image = pca(self.dataset)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()


    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path): 
                if self.tabWidget.currentIndex() == 0:
                    self.apply_face_detection(path)
                if self.tabWidget.currentIndex() == 1:
                    # self.recognition_image_view_widget.setImage(np.rot90(cv2.imread(path), k=-1))
                    self.apply_face_recognition(path)
                break


    def apply_face_detection(self, path):
        detected_faces_image = cv2.cvtColor(self.detect_and_draw_faces(cv2.imread(path)), cv2.COLOR_BGR2RGB)
        self.detected_image_view_widget.setImage(np.rot90(detected_faces_image, k=-1))

    def apply_face_recognition(self, path):
        test_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.recognition_image_view_widget.setImage(np.rot90(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), k=-1))
        
        self.recognition_label.setText("Processing....")
        self.recognition_label.setStyleSheet(f"background-color: {QColor(94, 92, 100).name()}; color: white; \
                                                 border-radius: 10px; width: 90%; Height: 40%;")
        self.recognition_label.setAlignment(Qt.AlignCenter)

        QCoreApplication.processEvents()

        recognized_class = recognize_face(test_img, self.eigen_faces, self.centered_data, self.mean_image, self.class_labels)

        if recognized_class in ['Unknown', 'Not A Face']:
            background_color = QColor(182, 13, 13)
        else:
            background_color = QColor(4, 175, 112)
        
        self.recognition_label.setText(recognized_class.split("\\")[-1])
        self.recognition_label.setStyleSheet(f"background-color: {background_color.name()}; color: white; \
                                                 border-radius: 10px; width: 90%; Height: 40%;")
        self.recognition_label.setAlignment(Qt.AlignCenter)
        QCoreApplication.processEvents()
        # print(f"Recognized Name: {recognized_class}")

    def detect_and_draw_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if len(image.shape) == 2:
            gray_image = image
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=20)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(30,16,166), thickness=3)

        return image

    def set_view_widget_settings(self, container):
        container.setBackground((14, 17, 23))
        container.setAspectLocked(True)
        container.hideAxis('left')
        container.hideAxis('bottom')

    def link_view_widgets(self):
        self.detected_image_view_widget, self.recognition_image_view_widget = pg.ImageItem(), pg.ImageItem()
        self.image_detection_widget.addItem(self.detected_image_view_widget)
        self.recognition_image_widget.addItem(self.recognition_image_view_widget)

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
