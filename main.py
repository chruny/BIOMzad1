import os
import numpy as np
import cv2
import csv
import math
from random import randint


# Created By Martin Kranec
#

class CSVLine():
    image_name = ''
    center_x_1 = ''
    center_y_1 = ''
    polomer_1 = ''
    center_x_2 = ''
    center_y_2 = ''
    polomer_2 = ''
    center_x_3 = ''
    center_y_3 = ''
    polomer_3 = ''
    center_x_4 = ''
    center_y_4 = ''
    polomer_4 = ''

    def __init__(self):
        pass

    def constructor(self, image_name, center_x_1, center_y_1, polomer_1, center_x_2, center_y_2, polomer_2, center_x_3,
                    center_y_3, polomer_3, center_x_4, center_y_4, polomer_4, ):
        self.image_name = image_name
        self.center_x_1 = center_x_1
        self.center_y_1 = center_y_1
        self.polomer_1 = polomer_1

        self.center_x_2 = center_x_2
        self.center_y_2 = center_y_2
        self.polomer_2 = polomer_2

        self.center_x_3 = center_x_3
        self.center_y_3 = center_y_3
        self.polomer_3 = polomer_3

        self.center_x_4 = center_x_4
        self.center_y_4 = center_y_4
        self.polomer_4 = polomer_4


pupils = []
iris = []
images = []
paths = []
csv_file = {}


def find_by_path(path):
    return csv_file[path]


def show(image_1, image_2, title):
    cv2.imshow(title, np.hstack([image_1, image_2]))
    cv2.waitKey()


def save(image, title):
    cv2.imwrite('new/'+str(randint(0,100))+'.jpg', image)

def get_common_value_from_list(list_of_values):
    average = np.average(list_of_values)
    for i in range(0, len(list_of_values)):
        list_of_values[i] = abs(average - list_of_values[i])
    index = list_of_values.index(min(list_of_values))
    print(index)
    return index


def statistical_edit_of_image(image, path):
    image_copy = image.copy()
    median_blur = cv2.medianBlur(image, 9)
    equaliz_median = cv2.equalizeHist(median_blur)

    gausian_blur = cv2.GaussianBlur(image, (5, 5), 1.7)
    equaliz_gauss = cv2.equalizeHist(gausian_blur)

    ret, thresh = cv2.threshold(equaliz_gauss, 50, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(equaliz_median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # show(image, thresh, 'threshold2')
    canny = cv2.Canny(thresh, 100, 200)
    # show(image, canny, 'canny1')
    canny2 = cv2.Canny(equaliz_median, 100, 200)
    # show(image, canny2, 'canny2')
    canny3 = cv2.Canny(median_blur, 100, 200)
    csv_file_new = CSVLine()
    csv_file_new = write_circles_pupil(median_blur, image, path, csv_file_new)
    csv_file_new = write_circles_iris(equaliz_gauss, image, path, csv_file_new)
    csv_file_new = write_circles_top_lid(canny, image, path, csv_file_new)
    csv_file_new = write_circles_bottom_lid(equaliz_gauss, image, path, csv_file_new)
    # show(image_copy, image, path)
    save(image, path)
    # write_circles(canny, image, path,is_pupil=True)


def detect_all_pupils():
    for i in range(0, len(images)):
        image = images[i]
        path = paths[i]
        statistical_edit_of_image(image, path)


# ------------------LOADING AND OTHER FUNCTIONS---------------

def write_circles_pupil(image, image2, path, csv_file_new):
    hough = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 10, param1=110, param2=1, minRadius=0, maxRadius=0)
    if hough is not None:

        for i in hough[0, :]:
            cv2.circle(image2, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(image2, (i[0], i[1]), 2, (0, 0, 255), 3)
            # show(image, image2, 'Pupil: ' + path)
            csv_file_new.image_name = path
            csv_file_new.center_x_1 = i[0]
            csv_file_new.center_y_1 = i[1]
            csv_file_new.polomer_1 = i[2]
            return csv_file_new


def write_circles_iris(image1, image2, path, csv_file_new):
    iris_objects = []
    iris_distances = []
    hough = cv2.HoughCircles(image1, cv2.HOUGH_GRADIENT, 2, 10, param1=100, param2=1,
                             minRadius=int(csv_file_new.polomer_1 * 1.7), maxRadius=int(csv_file_new.polomer_1 * 3.8))
    if hough is not None:
        for i in hough[0, :]:
            if (csv_file_new.polomer_1 * 3.5) > i[2] > (csv_file_new.polomer_1 * 1.7):
                iris_objects.append([i[0], i[1], i[2]])
                iris_distances.append(
                    distance_of_two_points(i[0], i[1], csv_file_new.center_x_1, csv_file_new.center_y_1))
        if len(iris_distances) > 0:
            # index = iris_distances.index(statistics.mode(iris_distances))
            index = iris_distances.index(min(iris_distances))
            # index = iris_distances.index(np.mean(iris_distances))
            csv_file_new.center_x_2 = iris_objects[index][0]
            csv_file_new.center_y_2 = iris_objects[index][1]
            csv_file_new.polomer_2 = iris_objects[index][2]
            cv2.circle(image2, (iris_objects[index][0], iris_objects[index][1]), iris_objects[index][2], (0, 255, 0), 2)
            # show(image1, image2, 'Iris: ' + path)
    return csv_file_new


def write_circles_top_lid(image1, image2, path, csv_file_new):
    min_radius = int(3 * csv_file_new.polomer_1)
    max_radius = int(5.8 * csv_file_new.polomer_1)
    iris_objects = []
    iris_distances = []
    hough = cv2.HoughCircles(image1, cv2.HOUGH_GRADIENT, 1, 6, param1=150, param2=5, minRadius=min_radius,
                             maxRadius=max_radius)
    if hough is not None:
        for i in hough[0, :]:
            iris_objects.append([i[0], i[1], i[2]])
            iris_distances.append(
                distance_of_two_points(i[0], i[1], csv_file_new.center_x_1, csv_file_new.center_y_1))
        if len(iris_distances) > 0:
            index = get_common_value_from_list(iris_distances)
            csv_file_new.center_x_2 = iris_objects[index][0]
            csv_file_new.center_y_2 = iris_objects[index][1]
            csv_file_new.polomer_2 = iris_objects[index][2]
            cv2.circle(image2, (iris_objects[index][0], iris_objects[index][1]), iris_objects[index][2], (0, 255, 0), 2)
            # show(image1, image2, 'Top lid ' + path)
    return csv_file_new


def write_circles_bottom_lid(image1, image2, path, csv_file_new):
    min_radius = int(4 * csv_file_new.polomer_1)
    max_radius = int(5.5 * csv_file_new.polomer_1)
    iris_objects = []
    iris_distances = []
    hough = cv2.HoughCircles(image1, cv2.HOUGH_GRADIENT, 1, 6, param1=150, param2=5, minRadius=min_radius,
                             maxRadius=max_radius)
    if hough is not None:
        for i in hough[0, :]:
            iris_objects.append([i[0], i[1], i[2]])
            iris_distances.append(
                distance_of_two_points(i[0], i[1], csv_file_new.center_x_1, csv_file_new.center_y_1))
        if len(iris_distances) > 0:
            # TODO chyba v mean nevraca premiernu hodnotu ale priemer
            index = get_common_value_from_list(iris_distances)
            csv_file_new.center_x_3 = iris_objects[index][0]
            csv_file_new.center_y_3 = iris_objects[index][1]
            csv_file_new.polomer_3 = iris_objects[index][2]
            cv2.circle(image2, (iris_objects[index][0], iris_objects[index][1]), iris_objects[index][2], (0, 255, 0), 2)
            # show(image1, image2, 'Bottom Lid' + path)
    return csv_file_new


# ---------------------POMOCNE FUNKCIE---------------------------------

def distance_of_two_points(x1, y1, x2, y2):
    return math.sqrt((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2)


def convert_to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_image(path):
    img = cv2.imread(path)
    if img is not None:
        return img


def load_all_images():
    main_directory = 'iris'
    directories = os.listdir(main_directory)
    for directory in directories:
        if os.path.isdir(main_directory + '/' + directory):
            sub_dirs = os.listdir(main_directory + '/' + directory)
            for sub_dir in sub_dirs:
                files = os.listdir(main_directory + '/' + directory + '/' + sub_dir)
                for file in files:
                    if 'bmp' in file:
                        img = load_image(main_directory + '/' + directory + '/' + sub_dir + '/' + file)
                        img = convert_to_gray_scale(img)
                        images.append(img)
                        paths.append(directory + '/' + sub_dir + '/' + file)


def load_csv():
    with open('iris/iris_bounding_circles.csv') as file:
        reader = csv.reader(file, delimiter=',')
        for i, line in enumerate(reader):
            if i > 0:
                line_csv = CSVLine()
                line_csv.constructor(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7],
                                     line[8], line[9], line[10], line[11], line[12])
                csv_file[line[0]] = line_csv


def main():
    load_all_images()
    detect_all_pupils()
    load_csv()
    quit()


if __name__ == '__main__':
    main()
