import os
import numpy as np
import cv2

pupils = []
iris = []
images = []


def convert_to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_image(path):
    img = cv2.imread(path)
    if img is not None:
        return img


def detect_pupil(image):
    image_copy = image.copy()
    # hough = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.3, 800)
    hough = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 50, 100, 50, 300)
    if hough is not None:
        hough = np.round(hough[0, :]).astype("int")
        for (x, y, raggio) in hough:
            cv2.circle(image_copy, (x, y), raggio, (255, 0, 0), 4)
        cv2.imshow("Image test", np.hstack([image, image_copy]))
        cv2.waitKey()


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


def detect_all_pupils():
    for image in images:
        detect_pupil(image)


def main():
    load_all_images()
    print(len(images))
    detect_all_pupils()
    quit()


if __name__ == '__main__':
    main()
