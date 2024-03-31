import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
from image_processing import ImageProcessor
from typing import Tuple 

class Snake:
    def __init__(self, image, filepath, alpha, beta, gamma, iteration, N_points= 25):
        self.image = image
        self.filepath = filepath
        self.N_points = N_points
        self.height, self.width = self.image.shape
        self.X_center= self.width//2
        self.y_center = self.height//2
        self.radius = self.width // 2.2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iteration = iteration
        self.contour_x, self.contour_y, self.window = self.create_contour()
        self.external_energy = self.calculate_external_energy(self.image)
        # Direction of the chain code
        self.directions = {
            0:  "North",
            1:  "North east",
            2:  "East",
            3:  "South east",
            4:  "South",
            5:  "South west",
            6:  "West",
            7:  "North west",
        }
    # initialize the contour
    def create_contour(self):
        angles = np.linspace(0, 2*np.pi, self.N_points)
        x_coordinates = self.X_center + self.radius* np.cos(angles)
        y_coordinates = self.y_center + self.radius * np.sin(angles)
        contour_x = x_coordinates.astype(int)
        contour_y = y_coordinates.astype(int)
        window = self.GenerateWindowCoordinates(7)
        return contour_x,contour_y , window
    
    # this function creates the indecis of each window pixel 
    def GenerateWindowCoordinates(self,Size):
        # Generate List of All Possible Point Values Based on Size
        Points = list(range(-Size // 2 + 1, Size // 2 + 1))
        PointsList = [Points, Points]

        # Generates All Possible Coordinates Inside The Window
        Coordinates = list(itertools.product(*PointsList))
        return Coordinates

    def calculate_internal_energy( self , contour_x , contour_y ):
        JoinedXY = np.array((contour_x, contour_y))
        # list of list inner list represent the point(x,y) and the outer one represents all points
        Points = JoinedXY.T
        # Continuous  Energy
        PrevPoints = np.roll(Points, 1, axis=0)
        NextPoints = np.roll(Points, -1, axis=0)
        Displacements = NextPoints - Points
        square_x = Displacements[:, 0] ** 2 
        square_y =Displacements[:, 1] ** 2
        ContinuousEnergy =np.sum(square_x+square_y)
        # Curvature Energy
        CurvatureSeparated = PrevPoints - 2 * Points + NextPoints
        Curvature = (CurvatureSeparated[:, 0] ** 2 + CurvatureSeparated[:, 1] ** 2)
        CurvatureEnergy = np.sum(Curvature)
        return self.alpha * ContinuousEnergy + self.beta * CurvatureEnergy

    def calculate_external_energy(self, source):
        # imgg_instance = ImageProcessor(self.filepath)
        src = np.copy(source)
        # convert to gray scale if not already
        if len(src.shape) > 2:
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        else:
            gray = src
        # Apply Gaussian Filter to smooth the image
        Gaussian_image = cv2.GaussianBlur(gray, (5, 5), 0)
        gradient_x = cv2.Sobel(Gaussian_image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(Gaussian_image, cv2.CV_64F, 0, 1, ksize=5)
        # Compute the magnitude of the gradient
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        # Normalize the gradient magnitude to range [0, 1]
        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
        external_energy =  - self.gamma * gradient_magnitude
        return external_energy

    def update_contour(self , source: np.ndarray, contour_x: np.ndarray, contour_y: np.ndarray,
                external_energy: np.ndarray, window_coordinates: list) -> Tuple[np.ndarray, np.ndarray]:
        src = source

        contour_points = len(contour_x)  # returns scaler of 60
        for Point in range(contour_points):  # point takes the value from 0 to 59
            MinEnergy = np.inf
            TotalEnergy = 0
            NewX = None
            NewY = None
            print("Countour_x1", contour_x[Point])
            # this for loops aims to update the position of the contour point
            for Window in window_coordinates: # (x,y) window[0]--> 43
                # Create Temporary Contours With Point Shifted To A Coordinate
                cont_x = contour_x[Point].copy()
                cont_y =contour_y[Point].copy()
                contour_x[Point] = contour_x[Point] + Window[0]   # shape[1]--> x axis
                contour_y[Point] = contour_y[Point] + Window[1] 
                print("Countour_x2", contour_x[Point])
                # Calculate Energy At The New Point
                TotalEnergy = external_energy[contour_y[Point],contour_x[Point]] +self.calculate_internal_energy(contour_x,contour_y)
                print("TotalEnergy", TotalEnergy)
                intenisty_of_point = src[contour_x[Point],contour_y[Point]]
                intenisty_of_Next_point= src[contour_x[Point]+1,contour_y[Point]+1]
                differance= np.abs(intenisty_of_point-intenisty_of_Next_point)
                # Save The Point If It Has The Lowest Energy In The Window
                if (TotalEnergy < MinEnergy and differance < 20 ):
                    MinEnergy = TotalEnergy
                    NewX = contour_x[Point] 
                    NewY = contour_y[Point] 
                    print("MinEnergy", MinEnergy)

                    print("NewX", NewX)
                    print("NewY", NewY)
                else :
                    NewX= cont_x
                    NewY=cont_y
            # Shift The Point In The Contour To It's New Location With The Lowest Energy
                contour_x[Point] = NewX
                contour_y[Point] = NewY
            print("Final countour_x ", contour_x[Point])
        chain_code, dir_words = self.generate_chain_code(contour_x, contour_y)
        print("Chain Code:", chain_code)
        print("Chain Code Words:", dir_words)

        area = self.calculate_area(contour_x, contour_y)
        perimeter = self.calculate_perimeter(contour_x, contour_y)
        print("Contour Area:", area)
        print("Contour Perimeter:", perimeter)
        return contour_x, contour_y



    def calculate_area(self, contour_x, contour_y):
        # Shoelace formula
        area = 0.5 * np.abs(np.dot(contour_x, np.roll(contour_y, 1)) - np.dot(contour_y, np.roll(contour_x, 1)))
        return area

    def calculate_perimeter(self, contour_x, contour_y):
        perimeter = 0
        for i in range(len(contour_x) - 1):
            dx = contour_x[i + 1] - contour_x[i]
            dy = contour_y[i + 1] - contour_y[i]
            perimeter += np.sqrt(dx**2 + dy**2)
        # Add the distance between the last and first points to close the contour
        dx = contour_x[0] - contour_x[-1]
        dy = contour_y[0] - contour_y[-1]
        perimeter += np.sqrt(dx**2 + dy**2)
        return perimeter

    def generate_chain_code(self, contour_x, contour_y):
        chain_code_sequence = []
        chain_code_sequence_word = []

        prev_point = (contour_x[0], contour_y[0])
        for point in zip(contour_x[1:], contour_y[1:]):
            dx = point[0] - prev_point[0]
            dy = point[1] - prev_point[1]

            direction = self.get_direction(dx, dy)
            if direction is not None:
                chain_code_sequence.append(direction)
                chain_code_sequence_word.append(self.directions[direction])

            prev_point = point
        return chain_code_sequence, chain_code_sequence_word

    def get_direction(self, dx, dy):
        if dy < 0:
            if dx > 0:
                return 1  # Northeast
            elif dx < 0:
                return 7  # Northwest
            else:
                return 0  # North
        elif dy > 0:
            if dx > 0:
                return 3  # Southeast
            elif dx < 0:
                return 5  # Southwest
            else:
                return 4  # South
        else:
            if dx > 0:
                return 2  # East
            elif dx < 0:
                return 6  # West
        return None
    


    

    