from spellchecker import SpellChecker
from funcs import *
import cv2
from CNN_OCR import *


im = cv2.imread("imgs/sample.jpeg")

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

bin = local_thresholding(im_bw, 11, set_mean=70)
bin = 255 - bin

kernel = np.ones((1, 2), 'uint8')

par = segment_paragraph(bin)

lines = segment_line(par)

letras = LetersCNN()

letras.loadNN()

text = ""

for line in lines:
    words = segment_word(line)
    for word in words:
        if word.shape[1] >= 5:
            chars = segment_character(word)
            j = []
            for i in range(0, len(chars)):
                j.append(cv2.resize(chars[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC))
            npa = np.asarray(j, dtype=np.float32)

            npa2 = npa.reshape(npa.shape[0], npa.shape[1], npa.shape[2], 1)
            klk = letras.predict(npa2)

            for i, v in enumerate(klk):
                pred = alphabet[np.argmax(klk[i])]
                text += pred
            text += " "

print(text)

spanish = SpellChecker(language='es')

corrected_text = ""
for word in text.split(" "):
    corrected_text += spanish.correction(word)
    corrected_text += " "
    print(spanish.correction(word))
    print(spanish.candidates(word))
print(text)
print(corrected_text)