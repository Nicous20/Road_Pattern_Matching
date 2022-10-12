import cv2 as cv
import numpy as np
import argparse
import time


def pattern_matching(img_file, temp_file, method):
    print("Start......")
    time_start = time.process_time()
    # Read image
    img_show = cv.imread(img_file).astype(np.float32)
    img = cv.cvtColor(img_show, cv.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    ret, img = cv.threshold(img, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    img = np.array([i ^ 1 for i in img])
    H, W = img.shape
    mi = np.mean(img)

    # Read template image
    temp_show = cv.imread(temp_file).astype(np.float32)
    temp = cv.cvtColor(temp_show, cv.COLOR_BGR2GRAY)
    temp = temp.astype(np.uint8)
    ret, temp = cv.threshold(temp, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    temp = np.array([i ^ 1 for i in temp])
    pixel_num = np.sum(temp)
    Ht, Wt = temp.shape

    block_par = 3
    block_ver = 3
    block_list = []
    for i in range(block_ver):
        for j in range(block_par):
            block_list.append([[Ht // block_ver * i, Ht // block_ver * (i + 1)], [Wt // block_par * j, Wt // block_par * (j+1)]])

    v, _v = -1, -1
    idx = -1
    for i in range(len(block_list)):
        y0 = block_list[i][0][0]
        y1 = block_list[i][0][1]
        x0 = block_list[i][1][0]
        x1 = block_list[i][1][1]
        img_block = img[y0:y1, x0:x1]
        temp_block = temp[y0:y1, x0:x1]
        # mi = np.mean(img[y0:y1, x0:x1])
        mt = np.mean(temp_block)

        _v = np.sum((img_block - mi) * (temp_block - mt))
        _v /= (np.sqrt(np.sum((img_block - mi) ** 2)) * np.sqrt(np.sum((temp_block - mt) ** 2)))
        if _v > v and np.sum(temp_block) > pixel_num / 10:
            v = _v
            idx = i

    y0 = block_list[idx][0][0]
    y1 = block_list[idx][0][1]
    x0 = block_list[idx][1][0]
    x1 = block_list[idx][1][1]

    # mi = np.mean(img[y0:y1, x0:x1])
    temp_block = temp[y0:y1, x0:x1]
    mt = np.mean(temp)

    # Template matching
    i, j = -1, -1
    v, _v = -1, -1


    for y in range(max(-H//10, -y0), min(H//10, Ht - y1)):
        for x in range(max(-W//10, -x0), min(W//10, Wt - x1)):
            _v = np.sum((img[y0+y:y1+y, x0+x:x1+x] - mi) * (temp_block - mt))
            _v /= (np.sqrt(np.sum((img[y0+y:y1+y, x0+x:x1+x] - mi) ** 2)) * np.sqrt(np.sum((temp_block - mt) ** 2)))
            if _v > v:
                v = _v
                i, j = x, y

    mat_translation = np.float32([[1, 0, i], [0, 1, j]])
    dst = cv.warpAffine(temp_show, mat_translation, (i + Wt, j + Ht), borderValue=(255, 255, 255))
    dst = dst.astype(np.uint8)

    result = img_show.copy()
    result = result.astype(np.uint8)
    for x in range(i, min(i + Wt, W)):
        for y in range(j, min(j + Ht, H)):
            if list(dst[y][x]) != [255, 255, 255]:
                result[y][x] = dst[y][x]

    time_end = time.process_time()
    print("End......")
    print("Time =====> ", time_end - time_start)
    cv.imwrite("./data/result.jpg", result)
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

pattern_matching("./data/post_4.jpg", "./data/pre_4.jpg", "ZNCC")