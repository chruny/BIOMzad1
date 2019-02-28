import os
import numpy as np
import cv2

pupils = []
iris = []
images = []


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


def detect_pupil_2(image):
    median_blur = cv2.medianBlur(image, 9)
    equaliz_median = cv2.equalizeHist(median_blur)
    # ret, threshold = cv2.threshold(equaliz_gaussian, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(equaliz_median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # show(image, thresh, 'threshold2')
    canny = cv2.Canny(thresh, 100, 200)
    # show(image, canny, 'canny1')
    canny2 = cv2.Canny(equaliz_median, 100, 200)
    # show(image, canny2, 'canny2')
    canny3 = cv2.Canny(median_blur, 100, 200)
    write_circles(canny, image)


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


def detect_all_pupils():
    for image in images:
        # detect_pupil(image)
        # detect_edges(image)
        detect_pupil_2(image)


# ------------------LOADING AND OTHER FUNCTIONS---------------

def write_circles(image, image2):
    image_copy = image.copy()
    hough = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 50, 100, 50, 300)
    if hough is not None:
        hough = np.round(hough[0, :]).astype("int")
        for (x, y, raggio) in hough:
            cv2.circle(image2, (x, y), raggio, (255, 0, 0), 4)
            show(image, image2, 'Circles')


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


def main():
    load_all_images()
    print(len(images))
    detect_all_pupils()
    quit()


if __name__ == '__main__':
    main()
