import cv2
import matplotlib.pyplot as plt

def detect_and_draw_faces(image):
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=20)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(30,16,166), thickness=3)

    return image



# image_with_faces = detect_and_draw_faces(cv2.imread('./Images/face_detection_test_image1.jpg'))
# plt.imshow(cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB))
# plt.show()