import cv2
import numpy as np
import itertools
class Snake:
    def __init__(self, image , N_points = 60 ):
        self.image = image
        self.N_points = N_points
        self.height, self.width = self.image.shape
        self.X_center= self.width//2
        self.y_center = self.height//2
        self.radius = self.width // 2.2

    # initialize the contour
    def create_contour(self ):
        angles = np.linspace(0, 2*np.pi, self.N_points)
        x_coordinates = self.X_center + self.radius * np.cos(angles)
        y_coordinates = self.y_center +self.radius * np.sin(angles)
        contour_x = x_coordinates.astype(int)
        contour_y = y_coordinates.astype(int)
        return contour_x , contour_y
    # this function creates the indecis of each window pixel 
    def GenerateWindowCoordinates(Size):
        # Generate List of All Possible Point Values Based on Size
        Points = list(range(-Size // 2 + 1, Size // 2 + 1))
        PointsList = [Points, Points]

        # Generates All Possible Coordinates Inside The Window
        Coordinates = list(itertools.product(*PointsList))
        return Coordinates
    
    

    


    

    