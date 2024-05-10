import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, uic 
import sys
import pyqtgraph as pg
import numpy as np
import cv2


class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi(r'task5.ui', self)
        
        self.view_widgets = [self.image_detection_widget]
        
        for container in self.view_widgets:
            self.set_view_widget_settings(container)

        self.detection_browse_button.clicked.connect(self.browse_image)

        self.link_view_widgets()

        self.setAcceptDrops(True)

    def browse_image(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        initial_folder = os.path.join(script_directory, "Images")
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", initial_folder, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if path:
            self.apply_face_detection(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                drop_position = event.pos()
                image_widget_position = self.image_detection_widget.mapFromGlobal(drop_position)
                image_widget_rect = self.image_detection_widget.rect()
                if image_widget_rect.contains(image_widget_position):
                    self.apply_face_detection(path)
                break


    def apply_face_detection(self, path):
        detected_faces_image = cv2.cvtColor(self.detect_and_draw_faces(cv2.imread(path)), cv2.COLOR_BGR2RGB)
        self.detected_image_view_widget.setImage(np.rot90(detected_faces_image, k=-1))

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
        self.detected_image_view_widget, self.detection_original_image_widget = pg.ImageItem(), pg.ImageItem()
        self.image_detection_widget.addItem(self.detected_image_view_widget)
        # self.manipulated_image_1.addItem(self.filter_manipulated_image_view_widget)

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
