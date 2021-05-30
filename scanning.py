from funcs import *
import cv2

def segment_paragraph(img):
    vertical_lines = cv2.erode(img, np.ones([40, 1]))
    img = img - vertical_lines

    img = cv2.erode(img, np.ones([2, 2]))
    img = cv2.dilate(img, np.ones([1, 3]))

    horizontal_lines = cv2.erode(img, np.ones([1, 40]))
    img = img - horizontal_lines

    img = cv2.erode(img, np.ones([2, 2]))
    img = cv2.dilate(img, np.ones([3, 1]))

    morph = cv2.dilate(img, np.ones([1, 400]))
    morph = cv2.dilate(morph, np.ones([200, 1]))
    pad = 5
    morph = np.pad(morph, 5)
    c = cv2.Canny(morph, 0, 255)
    imprint(c)
    contours, hierarchy = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=sort_ar)
    par = contours[-1]

    x, y, w, h = cv2.boundingRect(par)
    res = np.zeros((h, w), dtype='uint8')
    for i in range(h):
        for j in range(w):
            res[i, j] = img[y + i - pad, x + j - pad]

    return res


def segment_line(par):
    morph = cv2.dilate(par, np.ones([1, 400]))
    morph = cv2.erode(morph, np.ones([15]))
    morph = cv2.dilate(morph, np.ones([15]))

    pad = 5
    morph = np.pad(morph, 5)
    c = cv2.Canny(morph, 0, 255)

    contours, hierarchy = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=sort_ar)

    res = []
    for con in contours:
        x, y, w, h = cv2.boundingRect(con)
        r = np.zeros((h, w), dtype='uint8')
        for i in range(h):
            for j in range(w):
                r[i, j] = par[y + i - pad, x + j - pad]
        res.append((r, x, y))
    return res


def segment_character(line):
    h = np.zeros(line.shape[1])
    for i, v in enumerate(line.T):
        h[i] = np.sum(v)
    aux = h.copy()
    aux[aux == 0] = np.inf
    m = min(aux)
    letterLocations = np.ones(line.shape[1])
    for i, v in enumerate(h):
        if v < m:
            letterLocations[i] = 0

    d = np.diff(letterLocations)
    print(letterLocations)

    startingColumns = np.where(d > 0)[0]
    endingColumns = np.where(d < 0)[0]
    ret = []
    for i in range(len(startingColumns)):
        subImage = line[:, startingColumns[i]:endingColumns[i]]
        subImage = treat_character(subImage)
        imprint(subImage)
        ret.append(subImage)
    return ret


def treat_character(char):
    char = np.pad(char, 20)
    kernel = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0]], dtype='uint8')

    char = cv2.morphologyEx(char, cv2.MORPH_CLOSE, kernel)
    mask = char > 0
    char = char[np.ix_(mask.any(1), mask.any(0))]
    char = np.pad(char, 3)
    return char


im = cv2.imread("imgs/img18.jpeg")

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

par = segment_paragraph(bin)
lines = segment_line(par)

line = lines[10][0]
imprint(line)
chars = segment_character(line)
j = []
for i in range(0, len(chars)):
    #chars2[i] = cv2.resize(chars[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    j.append(cv2.resize(chars[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC))
npa = np.asarray(j, dtype=np.float32)
for i in range(0, len(npa)):
    cv2.imwrite('imgslletres/letra'+str(i)+'.jpg', npa[i])
hola = 1
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
