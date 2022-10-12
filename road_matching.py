import cv2 as cv
import numpy as np
import argparse


def pattern_matching(img_file, temp_file, method):
    # Read image
    img_show = cv.imread(img_file).astype(np.float32)
    img = cv.cvtColor(img_show, cv.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    ret, img = cv.threshold(img, 1, 0, cv.THRESH_BINARY + cv.THRESH_OTSU)
    H, W = img.shape
    mi = np.mean(img)

    # Read template image
    temp_show = cv.imread(temp_file).astype(np.float32)
    temp = cv.cvtColor(temp_show, cv.COLOR_BGR2GRAY)
    temp = temp.astype(np.uint8)
    ret, temp = cv.threshold(temp, 1, 0, cv.THRESH_BINARY + cv.THRESH_OTSU)
    Ht, Wt = temp.shape
    mt = np.mean(temp)

    # Template matching
    i, j = -1, -1
    v, _v = -1, -1
    if method == 'SSD' or method == 'SAD':
        v, _v = H * W * 255, H * W * 255

    for y in range(H - Ht):
        for x in range(W - Wt):
            if method == 'SSD':
                _v = np.sum((img[y:y + Ht, x:x + Wt] - temp) ** 2)
            elif method == 'NCC':
                _v = np.sum(img[y:y + Ht, x:x + Wt] * temp)
                _v /= (np.sqrt(np.sum(img[y:y + Ht, x:x + Wt] ** 2)) * np.sqrt(np.sum(temp ** 2)))
            elif method == 'SAD':
                _v = np.sum(np.abs(img[y:y + Ht, x:x + Wt] - temp))
            elif method == 'ZNCC':
                _v = np.sum((img[y:y + Ht, x:x + Wt] - mi) * (temp - mt))
                _v /= (np.sqrt(np.sum((img[y:y + Ht, x:x + Wt] - mi) ** 2)) * np.sqrt(np.sum((temp - mt) ** 2)))

            if method == 'SSD' or method == 'SAD':
                if _v < v:
                    v = _v
                    i, j = x, y
            else:
                if _v > v:
                    v = _v
                    i, j = x, y

    mat_translation = np.float32([[1, 0, i], [0, 1, j]])
    dst = cv.warpAffine(temp_show, mat_translation, (i + Wt, j + Ht), borderValue=(255, 255, 255))
    dst = dst.astype(np.uint8)

    result = img_show.copy()
    result = result.astype(np.uint8)
    for x in range(i, i + Wt):
        for y in range(j, j + Ht):
            if list(dst[y][x]) != [255, 255, 255]:
                result[y][x] = dst[y][x]

    cv.imwrite("./data/result.jpg", result)
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pattern Matching')
    parser.add_argument('-i', '--img_file', type=str, required=True, metavar='', help='Img File Name')
    parser.add_argument('-t', '--temp_file', type=str, required=True, metavar='', help='Template File Name')
    parser.add_argument('-m', '--method', type=str, required=True, metavar='', help='Method : [SSD][NCC][SAD][ZNCC]')
    args = parser.parse_args()
    pattern_matching(args.img_file, args.temp_file, args.method)
    # pattern_matching("post_3.png", "pre_3.png", "ZNCC")
