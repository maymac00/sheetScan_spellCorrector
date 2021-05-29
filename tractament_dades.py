import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def fun_internet(path):
    image = cv2.imread(path)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    # show the original and scanned images
    print("STEP 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(orig, height=650))
    cv2.imshow("Scanned", imutils.resize(warped, height=650))
    cv2.waitKey(0)


def homografy(im):
    limits = np.array([[0, 0], [im.shape[1], 0], [0, im.shape[0]], [im.shape[1], im.shape[0]]])
    # print(limits)
    plt.figure(1)
    plt.imshow(im, 'gray')
    x = plt.ginput(4, show_clicks=True)
    x = np.array(x)
    # print(x)

    h, status = cv2.findHomography(limits, x)

    im_dst = cv2.warpPerspective(im, h, im.shape)
    return im_dst


def tractament(file_path):
    im = cv2.imread(file_path)
    # im = ResizeWithAspectRatio(im, height=600)

    im_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #im_bw = homografy(im_bw)

    plt.imshow(im_bw, 'gray')
    plt.show()

    mask = im_bw > 160
    thresh = np.zeros_like(im_bw)
    thresh[mask] = 255

    cv2.imshow('image', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


fun_internet('imgs/img12.jpeg')