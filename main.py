
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QSlider,QHBoxLayout , QLabel
from PyQt5 import QtWidgets, QtCore, uic 
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
import os
import sys
import bisect
import pyqtgraph as pg
import numpy as np


    
class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Load the UI Page
        uic.loadUi(r'task1.ui', self)
        
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()