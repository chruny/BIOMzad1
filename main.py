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
# csv_file = {}
many_directories = []
actual_directory = ''


# pred_csv_line = {}


def define_parameter_top_lid():
    if actual_directory in ['002', '004']:
        return 3, 4
    return 4.2, 5.8


def define_parameter_bottom_lid():
    # 4.2 5.8
    if actual_directory in ['010']:
        return 3, 5.8
    if actual_directory in ['004']:
        return 4.25, 5.5
    else:
        return 4.3, 5.8


def find_by_path(path):
    return csv_file[path]


def show(image_1, image_2, title):
    cv2.imshow(title, np.hstack([image_1, image_2]))
    cv2.waitKey()


def save(image, title):
    cv2.imwrite('new/' + actual_directory + '/' + str(randint(0, 100)) + '.jpg', image)


def get_common_value_from_list(list_of_values):
    average = np.mean(list_of_values)
    for i in range(0, len(list_of_values)):
        list_of_values[i] = abs(average - list_of_values[i])
    index = list_of_values.index(min(list_of_values))
    return index


def statistical_edit_of_image(image, path):
    image_copy = image.copy()

    median_blur = cv2.medianBlur(image, 9)
    equaliz_median = cv2.equalizeHist(median_blur)

    equalize_hist = cv2.equalizeHist(image_copy)
    ret, thresh = cv2.threshold(equalize_hist, 50, 255, cv2.THRESH_BINARY)

    gausian_blur = cv2.GaussianBlur(image, (5, 5), 1.7)
    equaliz_gauss = cv2.equalizeHist(gausian_blur)

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
    csv_file_new.image_name = path
    return csv_file_new
    # write_circles(canny, image, path,is_pupil=True)


def detect_all_pupils():
    # TODO
    pred_csv_line = {}
    for i in range(0, len(images)):
        image = images[i]
        path = paths[i]
        actual_directory = many_directories[i]
        csv_file_new = statistical_edit_of_image(image, path)
        pred_csv_line[csv_file_new.image_name] = csv_file_new
    return pred_csv_line


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
    minimal, maximal = define_parameter_top_lid()
    min_radius = int(minimal * csv_file_new.polomer_1)
    max_radius = int(maximal * csv_file_new.polomer_1)
    iris_objects = []
    iris_distances = []
    x_offset = 15
    hough = cv2.HoughCircles(image1, cv2.HOUGH_GRADIENT, 1, 6, param1=150, param2=5, minRadius=min_radius,
                             maxRadius=max_radius)
    if hough is not None:
        for i in hough[0, :]:
            if i[1] > csv_file_new.center_y_1 + 110 and abs(i[0] - csv_file_new.center_x_1) < x_offset:
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
    minimal, maximal = define_parameter_bottom_lid()
    min_radius = int(minimal * csv_file_new.polomer_1)
    max_radius = int(maximal * csv_file_new.polomer_1)
    iris_objects = []
    iris_distances = []
    x_offset = 15
    hough = cv2.HoughCircles(image1, cv2.HOUGH_GRADIENT, 1, 6, param1=150, param2=5, minRadius=min_radius,
                             maxRadius=max_radius)
    if hough is not None:
        for i in hough[0, :]:
            if i[1] < csv_file_new.center_y_1 - 75 and abs(i[0] - csv_file_new.center_x_1) < x_offset:
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


# ---------------------INTERSECTION OVER UNION-------------------------

def intersection_over_union(csv_dict, pred_dict):
    print()
    for key in csv_dict.keys():
        # for key in range(0, len(pred_dict)):
        #     if csv_dict[key].image_name == pred_dict[key].image_name:
        pupil_per = get_overlap_in_percent(pred_dict[key].center_x_1, pred_dict[key].center_y_1,
                                           pred_dict[key].polomer_1,
                                           csv_dict[key].center_x_1, csv_dict[key].center_y_1, csv_dict[key].polomer_1)
        iris_per = get_overlap_in_percent(pred_dict[key].center_x_2, pred_dict[key].center_y_2,
                                          pred_dict[key].polomer_2,
                                          csv_dict[key].center_x_2, csv_dict[key].center_y_2, csv_dict[key].polomer_2)
        top_lid_per = get_overlap_in_percent(pred_dict[key].center_x_3, pred_dict[key].center_y_3,
                                             pred_dict[key].polomer_3,
                                             csv_dict[key].center_x_3, csv_dict[key].center_y_3,
                                             csv_dict[key].polomer_3)
        bot_lid_per = get_overlap_in_percent(pred_dict[key].center_x_4, pred_dict[key].center_y_4,
                                             pred_dict[key].polomer_4,
                                             csv_dict[key].center_x_4, csv_dict[key].center_y_4,
                                             csv_dict[key].polomer_4)
        total_per = (pupil_per + iris_per + top_lid_per + bot_lid_per) / 4
        print(csv_dict[key].image_name, '=', total_per)


def get_overlap_in_percent(x_pred, y_pred, polomer_pred, x_csv, y_csv, polomer_csv):
    distance_of_centers = distance_of_two_points(x_pred, y_pred, x_csv, y_csv)
    if distance_of_centers == abs(float(polomer_pred) - float(polomer_csv)):
        return 1
    elif distance_of_centers >= float(polomer_csv) + float(polomer_pred):
        total_area_1 = np.pi * float(polomer_csv) ** 2
        total_area_2 = np.pi * float(polomer_pred) ** 2
        if total_area_1 < total_area_2:
            return total_area_1 / total_area_2
        else:
            return total_area_2 / total_area_1
    else:
        polomer_pred_2, polomer_csv_2, distance_of_centers_2 = float(polomer_pred) ** 2, float(polomer_csv) ** 2, distance_of_centers ** 2
        alpha = np.arccos(
            (distance_of_centers_2 + polomer_csv_2 - polomer_pred_2) / (2 * distance_of_centers * float(polomer_csv)))
        beta = np.arccos(
            (distance_of_centers_2 + polomer_pred_2 - polomer_csv_2) / (2 * distance_of_centers * float(polomer_pred)))
        overlap_area = polomer_csv_2 * alpha * polomer_pred_2 * beta - 0.5 * (
                polomer_csv_2 * np.sin(2 * alpha) + polomer_pred_2 * np.sin(2 * beta))
        total_area_1 = np.pi * float(polomer_pred) ** 2
        total_area_2 = np.pi * float(polomer_csv) ** 2
        union_area = (total_area_1 - overlap_area) + total_area_2
        return overlap_area / union_area

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
                        many_directories.append(directory)
                        paths.append(directory + '/' + sub_dir + '/' + file)


def load_csv():
    csv_file = {}
    with open('iris/iris_bounding_circles.csv') as file:
        reader = csv.reader(file, delimiter=',')
        for i, line in enumerate(reader):
            if i > 0:
                line_csv = CSVLine()
                line_tmp = line[0].replace('_n2', '')
                line_csv.constructor(line_tmp, line[1], line[2], line[3], line[4], line[5], line[6],
                                     line[7],
                                     line[8], line[9], line[10], line[11], line[12])
                csv_file[line_tmp] = line_csv
    return csv_file


def main():
    print('Loading')
    load_all_images()
    print('Loading CSV')
    file_csv_lines = load_csv()
    print('Detecting')
    pred_csv_lines = detect_all_pupils()
    print('IOU')
    intersection_over_union(file_csv_lines, pred_csv_lines)
    quit()


if __name__ == '__main__':
    main()
