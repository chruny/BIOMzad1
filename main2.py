import os

import cv2


# src_gray: Input image (grayscale)
# circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
# CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
# dp = 1: The inverse ratio of resolution
# min_dist = src_gray.rows/8: Minimum distance between detected centers
# param_1 = 200: Upper threshold for the internal Canny edge detector
# param_2 = 100*: Threshold for center detection.
# min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
# max_radius = 0: Maximum radius to be detected. If unknown, put zero as default


def show(image, description="desc"):
    cv2.imshow(description, image)
    cv2.waitKey()


def konstanta(cast, folder):
    if cast == "dviecko":
        min = 4.3
        max = 5.2
        if folder in ["010"]:
            min = 3
        if folder in ["004"]:
            min = 4.25
            max = 5.5
    if cast == "hviecko":
        min = 4.2
        max = 5.8
        if folder in ["002", "004"]:
            min = 3
            max = 4

    return min, max


def wcircles(image=None, center_dist=10, minr=0, maxr=0, centerX=0, centerY=0, param1=110, param2=1,
             printToClassic=False, descr="the Popis", is_zornicka=False, xodchylka=10, yodchylka=13):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, center_dist, param1=param1, param2=param2, minRadius=minr,
                               maxRadius=maxr)
    retObj = {
        'x': 0,
        'y': 0,
        'r': 0,
    }
    circleFound = False
    circleDraw = False
    bestObj = {
        'x': 0,
        'y': 0,
        'r': 0,
        'pocet': 0,
    }
    if printToClassic is True:
        image = imageClassic
    if circles is not None:
        circleFound = True
        for i in circles[0, :]:
            if is_zornicka:
                circleDraw = True
                # draw the outer circle
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
                retObj['x'] = i[0]
                retObj['y'] = i[1]
                retObj['r'] = i[2]
                break
            else:
                if abs(centerX - i[0]) < xodchylka and abs(centerY - i[1]) < yodchylka:
                    circleDraw = True
                    bestObj['x'] += i[0]
                    bestObj['y'] += i[1]
                    bestObj['r'] += i[2]
                    bestObj['pocet'] += 1

    if not is_zornicka:
        if bestObj['pocet'] == 0:
            return retObj, circleFound, circleDraw

        bestObj['x'] = int(bestObj['x'] / bestObj['pocet'])
        bestObj['y'] = int(bestObj['y'] / bestObj['pocet'])
        bestObj['r'] = int(bestObj['r'] / bestObj['pocet'])
        cv2.circle(image, (bestObj['x'], bestObj['y']), bestObj['r'], (0, 255, 0), 2)
        cv2.circle(image, (bestObj['x'], bestObj['y']), 2, (0, 0, 255), 3)

        retObj['x'] = bestObj['x']
        retObj['y'] = bestObj['y']
        retObj['r'] = bestObj['r']
    if not printToClassic and circleDraw:
        show(image, descr + pathus)
    return retObj, circleFound, circleDraw


ThePrintus = True
root_folder = 'iris'
folders = os.listdir(root_folder)
for folder in folders:
    # if folder not in ["005","002", "003", "004", "005", "006"]:
    # if folder not in ["002"]:
    #         continue
    subfolders = os.listdir(root_folder + '/' + folder)
    for fold in subfolders:
        subsubf = os.listdir(root_folder + '/' + folder + '/' + fold)
        for eye_file in subsubf:
            if eye_file != "Thumbs.db":
                pathus = root_folder + '/' + folder + '/' + fold + '/' + eye_file

                imageClassic = cv2.imread(pathus)
                img = cv2.imread(pathus, 0)

                medianBlur = cv2.medianBlur(img, 9)
                gausianBlur = cv2.GaussianBlur(img, (5, 5), 1.7)

                equalizeHist = cv2.equalizeHist(img)
                gausianBlur = cv2.GaussianBlur(img, (5, 5), 1.7)
                equalizer_gausian = cv2.equalizeHist(gausianBlur)

                median_equalizer = cv2.equalizeHist(medianBlur)

                ret, thresh = cv2.threshold(equalizeHist, 50, 255, cv2.THRESH_BINARY)
                canny = cv2.Canny(thresh, 100, 200)

                canny2 = cv2.Canny(equalizeHist, 100, 200)
                canny3 = cv2.Canny(median_equalizer, 100, 200)

                cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                tmp = cv2.GaussianBlur(img, (5, 5), 1)
                tmp = cv2.equalizeHist(tmp)
                # ***************************************************** Zrenicka **********************************
                zreObj, the_circle_bool, the_draw_bool = wcircles(image=medianBlur, is_zornicka=True,
                                                                  descr="Zrenicka" + pathus, printToClassic=ThePrintus)
                print("***** STATISTIKA", pathus)

                # ***************************************************** Duhovka **********************************
                min_r = int(round(zreObj['r'] * 1.9))
                max_r = int(round(zreObj['r'] * 3.8))
                if folder in ["003"]:
                    min_r = min_r = int(round(zreObj['r'] * 2.4))
                newx = zreObj['x']
                newy = zreObj['y']
                xodchyl = 15
                yodchyl = 30
                it = 1
                while True:
                    duhovkaObj, the_circle_bool, the_draw_bool = wcircles(image=equalizer_gausian, minr=min_r,
                                                                          maxr=max_r, param1=100,
                                                                          param2=1, centerX=newx, centerY=newy,
                                                                          center_dist=2, descr="Duhovka" + pathus,
                                                                          xodchylka=xodchyl, yodchylka=yodchyl,
                                                                          printToClassic=ThePrintus)
                    if the_circle_bool and the_draw_bool:
                        break
                    if not the_draw_bool:  # ak nepreslo filtrom
                        xodchyl += it
                        yodchyl += it
                    if not the_circle_bool:  # ak nenaslo kruh
                        min_r += - it
                        max_r += it
                print(min_r, max_r)

                #
                # ***************************************************** horne viecko **********************************
                min_ = 4.2
                max_ = 5.8
                min_r_posun, max_r_posun = konstanta("hviecko", folder)
                min_r = int(round(zreObj['r'] * min_r_posun))
                max_r = int(round(zreObj['r'] * max_r_posun))
                newx = zreObj['x']
                newy = zreObj['y'] + 100
                xodchyl = 10
                yodchyl = 15

                it = 1
                while True:
                    hvieckoObj, the_circle_bool, the_draw_bool = wcircles(image=canny, minr=min_r, maxr=max_r,
                                                                          param1=150, param2=5, centerX=newx,
                                                                          centerY=newy, xodchylka=xodchyl,
                                                                          yodchylka=yodchyl,
                                                                          center_dist=6, descr="HorneViecko",
                                                                          printToClassic=ThePrintus)
                    if the_circle_bool and the_draw_bool:
                        break
                    if not the_draw_bool:  # ak nepreslo filtrom
                        xodchyl += it
                        yodchyl += it + it
                    if not the_circle_bool:  # ak nenaslo kruh
                        min_r += - it
                        max_r += it
                # ***************************************************** dolne viecko **********************************
                min_ = 4.3
                max_ = 5.2
                min_r_posun, max_r_posun = konstanta("dviecko", folder)
                min_r = int(round(zreObj['r'] * min_r_posun))
                max_r = int(round(zreObj['r'] * max_r_posun))
                newx = zreObj['x']
                newy = zreObj['y'] - 75
                xodchyl = 13
                yodchyl = 35
                if folder in ["002"]:
                    newy += -25
                print(min_r, max_r)
                while True:
                    if folder in ["004"]:
                        filterChoose = img
                    else:
                        filterChoose = equalizer_gausian
                    dvieckoObj, the_circle_bool, the_draw_bool = wcircles(image=filterChoose, minr=min_r, maxr=max_r,
                                                                          param1=150, param2=5, centerX=newx,
                                                                          centerY=newy, xodchylka=xodchyl,
                                                                          yodchylka=yodchyl,
                                                                          center_dist=6, descr="DolneViecko",
                                                                          printToClassic=True)
                    if the_circle_bool and the_draw_bool:
                        break
                    if not the_draw_bool:  # ak nepreslo filtrom
                        xodchyl += 1
                        print("************************************************************************")

                    if not the_circle_bool:  # ak nenaslo kruh
                        min_r += - 1
                        max_r += 1

                print(min_r, max_r)
                print("R zrenicka:", zreObj['r'], "Duhovka:", duhovkaObj['r'], "Horne viecko:", hvieckoObj['r'],
                      "Dolne viecko:", dvieckoObj['r'])
                print(duhovkaObj['r'] / zreObj['r'], hvieckoObj['r'] / zreObj['r'], dvieckoObj['r'] / zreObj['r'])
                print("X zrenicka:", zreObj['x'], "Duhovka:", duhovkaObj['x'], duhovkaObj['x'] - zreObj['x'],
                      "Horne viecko:", hvieckoObj['x'], "Dolne viecko:", dvieckoObj['x'], hvieckoObj['x'] - zreObj['x'],
                      dvieckoObj['x'] - zreObj['x'])

                print("Y zrenicka:", zreObj['y'], "Duhovka:", duhovkaObj['y'], duhovkaObj['y'] - zreObj['y'],
                      "Horne viecko:", hvieckoObj['y'], "Dolne viecko:", dvieckoObj['y'], hvieckoObj['y'] - zreObj['y'],
                      dvieckoObj['y'] + zreObj['y'])
                if ThePrintus:
                    show(imageClassic, pathus)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
