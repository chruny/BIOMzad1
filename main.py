import os
import numpy as np
import cv2
import csv

pupils = []
iris = []
images = []
paths = []
csv_file = {}
csv_file_new = ''


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


def find_by_path(path):
    return csv_file[path]


def show(image_1, image_2, title):
    cv2.imshow(title, np.hstack([image_1, image_2]))
    cv2.waitKey()


def detect_edges(image):
    cv2.imshow("Image test", np.hstack([image, ]))
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    laplac = cv2.Laplacian(image, cv2.CV_64F)
    cv2.imshow("Laplac", np.hstack([image, laplac]))
    cv2.imshow("SobelX", np.hstack([image, sobelx]))
    cv2.imshow("SobelY", np.hstack([image, sobely]))
    cv2.waitKey()


def statistical_edit_of_image(image, path):
    median_blur = cv2.medianBlur(image, 9)
    equaliz_median = cv2.equalizeHist(median_blur)

    gausian_blur = cv2.GaussianBlur(image, (5, 5), 1.7)
    equailiz_gauss = cv2.equalizeHist(gausian_blur)

    # ret, threshold = cv2.threshold(equaliz_gaussian, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(equaliz_median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # show(image, thresh, 'threshold2')
    canny = cv2.Canny(thresh, 100, 200)
    # show(image, canny, 'canny1')
    canny2 = cv2.Canny(equaliz_median, 100, 200)
    # show(image, canny2, 'canny2')
    canny3 = cv2.Canny(median_blur, 100, 200)
    write_circles_pupil(median_blur, image, path)
    # write_circles(canny, image, path,is_pupil=True)


def detect_all_pupils():
    for i in range(0, len(images)):
        image = images[i]
        path = paths[i]
        # detect_pupil(image)
        # detect_edges(image)
        statistical_edit_of_image(image, path)


# ------------------LOADING AND OTHER FUNCTIONS---------------

def write_circles_pupil(image, image2, path):
    image_copy = image.copy()
    hough = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 10, param1=110, param2=1, minRadius=0, maxRadius=0)
    # if hough is not None:
    #     hough = np.round(hough[0, :]).astype("int")
    #     for (x, y, raggio) in hough:
    #         cv2.circle(image2, (x, y), raggio, (255, 0, 0), 4)
    #         show(image, image2, 'Circles' + path)
    if hough is not None:

        for i in hough[0, :]:
            cv2.circle(image2, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(image2, (i[0], i[1]), 2, (0, 0, 255), 3)
            show(image, image2, 'Pupil: ' + path)
            csv_file_new = CSVLine()
            csv_file_new.image_name = path
            csv_file_new.center_x_1 = i[0]
            csv_file_new.center_y_1 = i[1]
            csv_file_new.polomer_1 = i[2]
            return


def write_circles_iris():
    print()


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
            print(main_directory, '/', directory)
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
        print()
        reader = csv.reader(file, delimiter=',')
        for i, line in enumerate(reader):
            if i > 0:
                line_csv = CSVLine()
                line_csv.constructor(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7],
                                     line[8], line[9], line[10], line[11], line[12])
                csv_file[line[0]] = line_csv


def main():
    load_all_images()
    # print(len(images))
    detect_all_pupils()
    load_csv()
    quit()


if __name__ == '__main__':
    main()
