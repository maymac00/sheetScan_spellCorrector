import cv2
import imutils
import numpy as np
from math import pi, floor, sqrt
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from skimage import transform


def img_divide(img, size, set_sd=20, set_mean=70):
    for column in range(0, img.shape[0], size):
        height = column + size
        for row in range(0, img.shape[1], size):
            width = row + size
            block = img[column:height, row:width]

            mean_block = np.mean(block)
            sd_block = np.std(block)
            if sd_block > set_sd:
                ret, block = cv2.threshold(block, 0, 255, cv2.THRESH_OTSU)
            elif sd_block < set_sd:
                if mean_block > set_mean:
                    block[:] = 255  # white
                else:
                    block[:] = 0  # black

            img[column:height, row:width] = block
    return img


def imprint(im):
    plt.figure(1)
    plt.imshow(im.astype('uint8'), 'gray')
    plt.show()


def local_threshold(image, block_size):
    k = gaussian_kernel(block_size, 2)
    thd = np.zeros(image.size, 'double')
    local_mean = fft_convolve2d(image, k)


def threshold(img, lowThresholdRatio=0.55, highThresholdRatio=1):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def fft_convolve2d(img, k):
    # x = np.fft.fft2(img)
    # y = np.fft.fft2(k, x.shape)
    # return abs(np.fft.ifft2(x * y))
    fr = np.fft.fft2(img)
    fr2 = np.fft.fft2(np.flipud(np.fliplr(k)), fr.shape)
    m, n = fr.shape
    cc = np.real(np.fft.ifft2(fr * fr2))
    return cc


def Ix(image):
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return fft_convolve2d(image, k)


def Iy(image):
    k = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return fft_convolve2d(image, k)


def gaussian_kernel(size, sigma):
    size = floor(int(size) / 2)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * pi * np.power(sigma, 2))
    g = np.exp(-((np.power(x, 2) + np.power(y, 2)) / (2.0 * np.power(sigma, 2)))) * normal
    return g


def sobel(img):
    im = fft_convolve2d(np.float32(img), gaussian_kernel(5, 1.4))
    # cv2.imshow('g', im.astype('uint8'))
    # cv2.waitKey(0)
    ix = Ix(im)
    iy = Iy(im)

    G = np.sqrt(np.power(ix, 2) + np.power(iy, 2))
    G = G / G.max() * 255
    theta = np.arctan2(iy, ix)

    return G, theta


def Canny(im, th1, th2):
    G, theta = sobel(im)

    res = non_max_suppression(G, theta)

    res, weak, strong = threshold(res, lowThresholdRatio=th1 / 255, highThresholdRatio=th2 / 255)
    return res.astype('uint8')


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


def homografy(image, points, which='lib'):
    if which == 'lib':
        warped = four_point_transform(image, points)
        return warped

    tform = getTransformMatrix(points, image.shape)
    invTform = np.linalg.inv(tform.params)
    shape = [0, 0]
    p1 = points[0]
    p2 = points[1]
    p3 = points[0]
    p4 = points[3]
    pad = 180
    lpad = 30
    rpad = 0
    shape[0] = int(sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)) + pad
    shape[1] = int(sqrt((p3[0] - p4[0]) ** 2 + (p3[1] - p4[1]) ** 2)) + pad
    shape.append(3)
    new = np.zeros(shape)

    for i, row in enumerate(new):
        for j, val in enumerate(row):
            src = np.array([[i], [j], [1]])
            res = np.dot(invTform, src)
            res = res / res[2]
            x = res[0]
            y = res[1]
            if rpad < x < new.shape[0] and lpad < y < new.shape[1]:
                new[i, j] = image[int(x[0]), int(y[0] - lpad)]

    return new


def getTransformMatrix(points, shp):
    obj = np.array([[0, 0], [0, shp[0]], [shp[1], shp[0]], [shp[1], 0]])
    tform = transform.estimate_transform('projective', np.array(points), np.array(obj))
    return tform


def sort_ar(ar):
    return ar.shape[0]
