from funcs import *
import cv2
from CNN_OCR import *


def segment_paragraph(img):
    vertical_lines = cv2.erode(img, np.ones([40, 1]))
    img = img - vertical_lines

    img = cv2.erode(img, np.ones([2, 2]))
    img = cv2.dilate(img, np.ones([1, 3]))

    horizontal_lines = cv2.erode(img, np.ones([1, 40]))
    img = img - horizontal_lines

    img = cv2.erode(img, np.ones([2, 2]))
    img = cv2.dilate(img, np.ones([3, 1]))

    morph = cv2.erode(img, np.ones([5, 5]))
    morph = cv2.dilate(morph, np.ones([5, 5]))

    morph = cv2.dilate(morph, np.ones([1, 400]))
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
    morph = cv2.erode(par, np.ones([15]))

    morph = cv2.dilate(morph, np.ones([1, 400]))
    morph = cv2.erode(morph, np.ones([15]))
    morph = cv2.dilate(morph, np.ones([20]))
    morph = cv2.dilate(morph, np.ones([20, 1]))

    pad = 5
    morph = np.pad(morph, 5)
    c = cv2.Canny(morph, 0, 255)
    imprint(c)
    contours, hierarchy = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=sort_position)
    res = []
    for con in contours:
        x, y, w, h = cv2.boundingRect(con)
        r = np.zeros((h, w), dtype='uint8')
        for i in range(h):
            for j in range(w):
                r[i, j] = par[y + i - pad, x + j - pad]
        res.append(r)
    return res


def segment_word(line):
    morph = cv2.erode(line, np.ones([8,5]))
    morph = cv2.dilate(morph, np.ones([8, 5]))
    imprint(morph)
    morph = cv2.dilate(morph, np.ones([1, 55]))
    imprint(morph)
    h = np.zeros(morph.shape[1])
    for i, v in enumerate(morph.T):
        h[i] = np.sum(v)
    aux = h.copy()
    aux[aux == 0] = np.inf
    m = min(aux)
    wordLocations = np.ones(morph.shape[1])
    for i, v in enumerate(h):
        if v < m:
            wordLocations[i] = 0

    d = np.diff(wordLocations)

    startingColumns = np.where(d > 0)[0]
    endingColumns = np.where(d < 0)[0]
    ret = []
    for i in range(len(startingColumns)):
        subImage = line[:, startingColumns[i]:endingColumns[i]]
        ret.append(subImage)
    return ret


def segment_character(word):
    h = np.zeros(word.shape[1])
    for i, v in enumerate(word.T):
        h[i] = np.sum(v)
    aux = h.copy()
    aux[aux == 0] = np.inf
    m = min(aux)
    letterLocations = np.ones(word.shape[1])
    for i, v in enumerate(h):
        if v < m:
            letterLocations[i] = 0

    d = np.diff(letterLocations)

    startingColumns = np.where(d > 0)[0]
    endingColumns = np.where(d < 0)[0]
    if endingColumns.shape[0] < startingColumns.shape[0]:
        endingColumns = np.append(endingColumns, startingColumns.shape[0]-1)
    if startingColumns.shape[0] < endingColumns.shape[0]:
        startingColumns = np.append(startingColumns, 0)
    ret = []
    for i in range(len(startingColumns)):
        subImage = word[:, startingColumns[i]:endingColumns[i]]
        subImage = treat_character(subImage)
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

letras = LetersCNN()

letras.loadNN()

str = ""

for line in lines:
    words = segment_word(line)
    imprint(line)
    for word in words:
        if word.shape[1] >= 5:
            imprint(word)
            chars = segment_character(word)
            j = []
            for i in range(0, len(chars)):
                j.append(cv2.resize(chars[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC))
            npa = np.asarray(j, dtype=np.float32)

            npa2 = npa.reshape(npa.shape[0], npa.shape[1], npa.shape[2], 1)
            klk = letras.predict(npa2)

            for i, v in enumerate(klk):
                pred = alphabet[np.argmax(klk[i])]
                str += pred
            str += " "

print(str)
"""
for i in range(0, len(npa)):
    cv2.imwrite('imgslletres/letra' + str(i) + '.jpg', npa[i])"""
