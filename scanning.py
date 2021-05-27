import cv2
import numpy as np
from math import pi, floor
import matplotlib.pyplot as plt


def imprint(im):
    plt.figure(1)
    plt.imshow(im.astype('uint8'), 'gray')
    plt.show()


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

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
    cc = cc.astype('uint8')

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


def Canny(im):
    im = fft_convolve2d(im, gaussian_kernel(65, 10))
    # cv2.imshow('g', im.astype('uint8'))
    # cv2.waitKey(0)
    ix = Ix(im)
    iy = Iy(im)

    G = np.sqrt(np.power(ix, 2) + np.power(iy, 2))
    G = G / G.max() * 255

    theta = np.arctan2(iy, ix)
    res = non_max_suppression(G, theta)
    res = res.astype('uint8')

    res = threshold(res)
    imprint(res)
    pass


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


def homografy(im):
    limits = np.array([[0, 0], [im.shape[1], 0], [0, im.shape[0]], [im.shape[1], im.shape[0]]])
    # print(limits)
    plt.figure(1)
    plt.imshow(im, 'gray')
    x = plt.ginput(4, show_clicks=True)
    x = np.array(x)
    # print(x)

    h, status = cv2.findHomography(x, limits)

    im_dst = cv2.warpPerspective(im, h, im.shape)
    return im_dst


def tractament(file_path):
    im = cv2.imread(file_path)
    im = ResizeWithAspectRatio(im, height=600)

    im_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im_bw = homografy(im_bw)

    plt.imshow(im_bw, 'gray')
    plt.show()

    mask = im_bw < 90
    thresh = np.zeros_like(im_bw)
    thresh[mask] = 255

    cv2.imshow('image', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# tractament("image_1.jpeg")
im = cv2.imread("image_1.jpeg")

im_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imprint(Canny(im_bw))
# Canny(im_bw)
