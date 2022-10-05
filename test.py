import numpy as np
import cv2 as cv

img_file = "pre.jpg"
img_show = cv.imread(img_file).astype(np.float32)
img = cv.cvtColor(img_show, cv.COLOR_BGR2GRAY)
img = img.astype(np.uint8)
ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("img", img)

cv.waitKey(0)
cv.destroyAllWindows()