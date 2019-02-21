import os
import numpy as np
import cv2

pupils = []
iris = []

class Pupil:

    center_x = ''
    center_y = ''
    radius = ''

    def __init__(self,center_x,center_y,radius):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

def convert_to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def load_image(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    print()


def detect_pupil(img):
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    _, t = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
    contours, _, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = cv2.HoughCircles(contours, cv2.HOUGH_GRADIENT, 2, img.shape[0] / 2)
    for l in c:
        for circle in l:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(img, center, radius, (0, 0, 0), thickness=-1)
            pupil = Pupil(center[0],center[1], center[2])
            pupils.append(pupil)


def detect_iris(img):
    _, t = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
    countours, _, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c = cv2.HoughCircles(countours, cv2.HOUGH_GRADIENT, 2, )

def load_all_images():
    files = os.listdir('iris/001/1/')
    path = 'iris/001/1/001_1_2.bmp'
    img = load_image(path)
    img = convert_to_gray_scale(img)
    detect_pupil(img)
    print(img)
    print(files)


def main():
    load_all_images()
    print()


if __name__ == '__main__':
    main()
