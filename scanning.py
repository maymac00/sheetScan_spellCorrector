from funcs import *


def segment_paragraph(img):
    r = img.shape[1]/ img.shape[0]
    vertical_lines = cv2.erode(img, np.ones([40, 1]))
    img = img-vertical_lines

    img = cv2.erode(img, np.ones([2, 2]))
    img = cv2.dilate(img, np.ones([1, 3]))

    horizontal_lines =  cv2.erode(img, np.ones([1, 40]))
    img = img - horizontal_lines

    img = cv2.erode(img, np.ones([2, 2]))
    img = cv2.dilate(img, np.ones([3, 1]))

    morph = cv2.dilate(img, np.ones([1, 400]))
    morph = cv2.dilate(morph, np.ones([80, 1]))
    pad = 5
    morph = np.pad(morph, 5)
    c = cv2.Canny(morph, 0, 255)
    imprint(morph)
    imprint(c)
    contours, hierarchy = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=sort_ar)

    for count in contours[-5:]:
        x, y, w, h = cv2.boundingRect(count)
        res = np.zeros((h, w), dtype='uint8')
        for i in range(h):
            for j in range(w):
                res[i, j] = img[y + i -pad, x + j-pad]
    return img

im = cv2.imread("imgs/img14.jpeg")

im_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
can = Canny(im_bw, 30, 255)

cnt = cv2.findContours(can.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnt)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
screenCnt = screenCnt.reshape(4, 2)
warped = homografy(im, screenCnt.reshape(4, 2))

im_bw = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

bin = img_divide(im_bw, 11, set_mean=70)
bin = 255 - bin

imprint(bin)

kernel = np.ones((1, 2), 'uint8')

segment_paragraph(bin)

"""
contours, hierarchy = cv2.findContours(can, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=sort_ar)

imprint(bin)
if len(contours) == 0:
    exit(0)

words = []
for c in contours:
    if c.shape[0] < 1200:
        words.append(c)

cv2.drawContours(warped, words, -1, (0, 255, 0), 3)
imprint(warped)
x, y, w, h = cv2.boundingRect(txt)

res = np.zeros((h, w), dtype='uint8')
for i in range(h):
    for j in range(w):
        res[i, j] = bin[y + i, x + j]
"""