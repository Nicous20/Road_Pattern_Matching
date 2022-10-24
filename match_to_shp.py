import cv2 as cv
import numpy as np
import argparse
import time


def pattern_matching(img_file, temp_file):
    # fix temp_file
    kernel = np.ones((5, 5), np.uint8)
    print("Start......")
    time_start = time.process_time()
    # Read image
    img_show = cv.imread(img_file).astype(np.float32)
    img = cv.cvtColor(img_show, cv.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    ret, img = cv.threshold(img, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # img = np.array([i ^ 1 for i in img])
    # img = cv.dilate(img, kernel)
    H, W = img.shape

    # Read template image
    temp_show = cv.imread(temp_file).astype(np.float32)
    temp = cv.cvtColor(temp_show, cv.COLOR_BGR2GRAY)
    temp = temp.astype(np.uint8)
    ret, temp = cv.threshold(temp, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # temp = np.array([i ^ 1 for i in temp])
    # temp = cv.dilate(temp, kernel)
    Ht, Wt = temp.shape

    block_par = 3
    block_ver = 3
    block_list = []
    for i in range(block_ver):
        for j in range(block_par):
            block_list.append([[Ht // block_ver * i, Ht // block_ver * (i + 1)], [Wt // block_par * j, Wt // block_par * (j + 1)]])

    # v, _v = -1, -1
    # idx = 0
    # for i in range(len(block_list)):
    #     y0 = block_list[i][0][0]
    #     y1 = block_list[i][0][1]
    #     x0 = block_list[i][1][0]
    #     x1 = block_list[i][1][1]
    #     img_block = img[y0:y1, x0:x1]
    #     temp_block = temp[y0:y1, x0:x1]
    #     # mi = np.mean(img[y0:y1, x0:x1])
    #     mt = np.mean(temp_block)
    #     if mt < 1e-6:
    #         continue
    #
    #     _v = np.sum((img_block - mi) * (temp_block - mt))
    #     _v /= (np.sqrt(np.sum((img_block - mi) ** 2)) * np.sqrt(np.sum((temp_block - mt) ** 2)))
    #     if _v > v and np.sum(temp_block) > pixel_num / 10:
    #         v = _v
    #         idx = i

    idx = 4
    y0 = block_list[idx][0][0]
    y1 = block_list[idx][0][1]
    x0 = block_list[idx][1][0]
    x1 = block_list[idx][1][1]

    cv.imshow("cut", temp_show[y0:y1, x0:x1].astype(np.uint8))
    cv.imwrite("data/pre_cut_auto.jpg", temp_show[y0:y1, x0:x1].astype(np.uint8))

    temp_block = temp[y0:y1, x0:x1]
    len_y = y1 - y0
    len_x = x1 - x0
    mi = np.mean(img)
    mt = np.mean(temp_block)

    # Template matching
    i, j = 0, 0
    v, _v = -1, -1
    v = np.sum((img[y0 + i:y1 + i, x0 + j:x1 + j] - mi) * (temp_block - mt))
    v /= (np.sqrt(np.sum((img[y0 + i:y1 + i, x0 + j:x1 + j] - mi) ** 2)) * np.sqrt(np.sum((temp_block - mt) ** 2)))
    step = 2
    for y in range(-len_y, len_y, step):
        for x in range(-len_x, len_x, step):
            y_ = y0 + y
            y__ = y1 + y
            x_ = x0 + x
            x__ = x1 + x

            _v = np.sum((img[y_:y__, x_:x__] - mi) * (temp_block - mt))
            _v /= (np.sqrt(np.sum((img[y_:y__, x_:x__] - mi) ** 2)) * np.sqrt(np.sum((temp_block - mt) ** 2)))
            if _v > v:
                v = _v
                i, j = y, x

    i_, j_ = i, j
    for y in range(-step, step):
        for x in range(-step, step):
            y_ = y0 + i_ + y
            y__ = y1 + i_ + y
            x_ = x0 + j_ + x
            x__ = x1 + j_ + x
            _v = np.sum((img[y_:y__, x_:x__] - mi) * (temp_block - mt))
            _v /= (np.sqrt(np.sum((img[y_:y__, x_:x__] - mi) ** 2)) * np.sqrt(np.sum((temp_block - mt) ** 2)))
            if _v > v:
                v = _v
                i, j = i_ + y, j_ + x


    i, j = j, i
    mat_translation = np.float32([[1, 0, i], [0, 1, j]])
    dst = cv.warpAffine(temp_show, mat_translation, (i + Wt, j + Ht), borderValue=(255, 255, 255))
    dst = dst.astype(np.uint8)
    Ht, Wt, C = dst.shape

    result = img_show.copy()
    result = result.astype(np.uint8)
    for x in range(0, min(Wt, W) - 1):
        for y in range(0, min(Ht, H) - 1):
            if list(dst[y][x]) != [255, 255, 255]:
                result[y][x] = dst[y][x]

    time_end = time.process_time()
    print("End......")
    print("Time =====> ", time_end - time_start)
    cv.imwrite("./data/result.jpg", result)
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()



img_pre = "2020陕西商洛山洪/灾前提取结果/centerline.shp"
img_post = "2020陕西商洛山洪/灾后提取结果/洛南灾后道路提取结果/洛南/centerline_洛南灾区.shp"

img_post = "./data/post.jpg"
img_pre = "./data/pre.jpg"

pattern_matching(img_post, img_pre)




