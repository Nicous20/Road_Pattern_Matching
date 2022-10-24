# from utils import CenterlineExtraction,imgPad,concat,connect
import copy
from math import sqrt
from distutils.archive_util import make_archive
import time
from math import log10

from pip import main
import numpy as np
# from osgeo import ogr, gdal
from tqdm import tqdm
import ctypes
import json
from ctypes import *
import cv2
import os
from platform import system
import multiprocessing
from simplification.cutil import (
    simplify_coords,
    simplify_coords_idx,
    simplify_coords_vw,
    simplify_coords_vw_idx,
    simplify_coords_vwp,
)

if system() == 'Windows':
    pDLL = WinDLL("./centerline.dll", winmode=0)
elif system() == 'Linux':
    pDLL = cdll.LoadLibrary("./libMySharedLib.so")
pDLL.CenterlineExtraction.restype = ctypes.c_uint
pDLL.CenterlineExtraction.argtypes = [ctypes.POINTER(ctypes.c_ubyte), c_int, c_int, c_int, c_int, ctypes.POINTER(
    ctypes.POINTER(c_float)), ctypes.POINTER(ctypes.POINTER(c_float)), ctypes.POINTER(ctypes.POINTER(c_uint))]
use_index = []


# if __name__ == '__main__':


def CenterlineExtractionProcess(mask, output_path):
    # if system() == 'Windows':
    #     pDLL = WinDLL("./centerline.dll",winmode=0)
    # elif system()=='Linux':
    #     pDLL = cdll.LoadLibrary("./libMySharedLib.so")
    # pDLL.CenterlineExtraction.restype=ctypes.c_uint
    # pDLL.CenterlineExtraction.argtypes=[ctypes.POINTER(ctypes.c_ubyte),c_int,c_int,c_int,c_int,ctypes.POINTER(ctypes.POINTER(c_float)),ctypes.POINTER(ctypes.POINTER(c_float)),ctypes.POINTER(ctypes.POINTER(c_uint))]
    # print(pDLL)
    tic = time.time()
    # dataset = gdal.Open("/nfs/ssdk/mask2.tif")
    # dataset = gdal.Open(input_path)
    # width = dataset.RasterXSize  # 图像宽度
    # height = dataset.RasterYSize  # 图像高度
    height, width = mask.shape
    print("开始读取")
    # img = dataset.ReadAsArray(0, 0, width, height).transpose((1,2,0))[:,:,0]
    img = mask
    toc = time.time()
    print("读取tif耗时：", toc - tic)
    # img = cv2.imread("lushan_65556_mask.tif", -1)
    print(img.shape)
    padSize = 1024
    pad = 100
    oriSize = padSize - (pad * 2)
    json_list = []
    for i in tqdm(range(0, 64506, oriSize)):  # y
        for j in range(0, 64506, oriSize):  # x
            tmp = imgPad((j, i), img, (oriSize, oriSize), pad)
            res = CenterlineExtraction(tmp, 0, 0, (j - pad, i - pad))
            tmp_json = {}
            tmp_json["x"] = j
            tmp_json["y"] = i
            tmp_json["data"] = res
            json_list.append(tmp_json)
            # filename="./output/"+"{}_{}_{}.json".format(j,i,pad)
            # with open(filename,'w') as file_obj:
            # json.dump(res,file_obj)
    geojson = concat(json_list, ori_size=oriSize)
    # geojson={}
    # with open("./concat.json",'r') as file_obj:
    #     geojson=json.load(file_obj)
    exclude = []
    tic2 = time.time()
    new_js = geojson
    print("copy json time:", time.time() - tic2)
    tic2 = time.time()
    use_index = np.arange(0, len(new_js["features"]), 1).tolist()
    print("gen list time:", time.time() - tic2)
    # use_index=[x for x in range(0,len(new_js["features"]))]
    index = 0
    for feature in tqdm(new_js["features"]):
        if (feature["pop"] > 0):
            try:
                use_index.remove(index)
            except:
                pass
            index += 1
            continue
        new_js["features"][index]["pop"] = 1
        connect(new_js, use_index, index)
        index += 1
    index = 0
    while (True):
        if (new_js["features"][index]["pop"] == 2):
            new_js["features"].pop(index)
        else:
            index += 1
        if (len(new_js["features"]) == index):
            break
    # filename="connect.json"
    output_path = output_path + "connect.json"
    print("end")
    toc = time.time()
    print("总耗时：", toc - tic)
    for index, js in enumerate(new_js["features"]):
        line = js["geometry"]["coordinates"]
        new_js["features"][index]["geometry"]["coordinates"] = simplify_coords_vw(
            line, 10)
    with open("./connect2.json", "w") as fp2:
        fp2.write(json.dumps(new_js))
    with open(output_path, 'w') as file_obj:
        json.dump(new_js, file_obj)

    # print(res)


def CenterlineExtraction(img, pruned, smooth, offset=(0, 0)):
    offsetX, offsetY = offset
    ox = pointer(c_float(0.))
    oy = pointer(c_float(0.))
    oi = pointer(c_uint(0))
    width = c_int(img.shape[1])
    height = c_int(img.shape[0])
    pruned = c_int(pruned)
    smooth = c_int(smooth)
    # img=cv2.imread("4.png",-1)
    img = np.array(img, dtype=np.uint8)[:, :]
    img_p = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    # 调用动态链接库函数
    nsegs = pDLL.CenterlineExtraction(
        img_p, width, height, pruned, smooth, pointer(ox), pointer(oy), pointer(oi))
    data = {}
    data['features'] = []
    for i in range(nsegs):
        f = {}
        f['id'] = i
        f['coordinates'] = []
        ssize = int(oi[i + 1]) - int(oi[i])
        for pi in range(int(oi[i]), int(oi[i + 1])):
            f['coordinates'].append([ox[pi] + offsetX, oy[pi] + offsetY])
            cv2.circle(img, (int(ox[pi]), int(oy[pi])), 1, 20)
        data['features'].append(f)
    return data
    filename = 'test.json'

def split(bool_line): # 一个布尔列表
    line_list=[]
    length=bool_line.shape[0]
    bool_sample=[False for i in range(length)]
    L=0
    R=0
    isRecord=False
    for index,p in enumerate(bool_line):
        if p and not isRecord:
            L=index
            isRecord=True
        if not p and isRecord:
            R=index
            isRecord=False
            bool_tmp=np.array(bool_sample)
            bool_tmp[L:R]=True
            line_list.append(bool_tmp)
        if p and index==length-1: #最后
            isRecord=False
            bool_tmp=np.array(bool_sample)
            bool_tmp[L:]=True
            line_list.append(bool_tmp)
    return line_list
def imgPad(startPoint, img, size, pad):
    padZero=50
    height, width = size
    offsetE = width
    offsetS = height  # 有效orisize
    paddE = pad
    paddS = pad
    x, y = startPoint
    if(x+width+pad >= img.shape[1]):
        if(x+width >= img.shape[1]):
            offsetE = img.shape[1]-x
            paddE = 0
        else:
            paddE = img.shape[1]-x-width
    if(y+height+pad >= img.shape[0]):
        if(y+height >= img.shape[0]):
            offsetS = img.shape[0]-y
            paddS = 0
        else:
            paddS = img.shape[0]-y-height
    tmp = np.zeros((pad * 2 + height, pad * 2 + width))
    tmp2 = np.zeros((pad * 2 + height, pad * 2 + width))
    # print("tmp shape:",tmp.shape)
    # print("x,y:",x,y," img shape:",img[y:y+height,x:x+width].shape)
    tmp[pad:pad + offsetS + paddS, pad:pad + offsetE +
        paddE] = img[y:y + offsetS + paddS, x:x + offsetE +
                     paddE]
    if (y != 0):
        tmp[:pad, pad:pad + offsetE + paddE] = img[y - pad:y, x:x + offsetE + paddE]
    if (x != 0):
        tmp[pad:pad + offsetS + paddS, :pad] = img[y:y + offsetS + paddS, x - pad:x]
    if (x != 0 and y != 0):
        tmp[:pad, :pad] = img[y - pad:y, x - pad:x]
    tmp2[padZero:-padZero,padZero:-padZero]=tmp[padZero:-padZero,padZero:-padZero]
    return tmp2


def concat(json_list, ori_size):
    geojson = {}
    geojson["type"] = "FeatureCollection"
    geojson["features"] = []
    total_num = 0
    line_id = 0
    for json_tmp in json_list:
        tmp = json_tmp["data"]
        # print(tmp)
        start_x, start_y = json_tmp["x"], json_tmp["y"]
        start_x = int(start_x)
        start_y = int(start_y)
        for js in tmp["features"]:
            f = {}
            f["pop"] = 0  # 0有链接 1无连接 2预备删除
            f["start"] = {}
            x = start_x // ori_size
            y = start_y // ori_size
            zero_num = 10 ** len(str(y))
            f["start"]["xy"] = [x * zero_num + (y)]
            f["start"]["connect"] = []
            f["type"] = "Feature"
            f["properties"] = {}
            f["properties"]["FID"] = 0
            f["geometry"] = {}
            f["geometry"]["type"] = "LineString"
            tmp_list = np.array(js["coordinates"])
            total_num += tmp_list.shape[0]
            a1 = (tmp_list[:, 0] >= start_x)
            if (False in a1):
                f["start"]["connect"].append((x - 1) * zero_num + y)
            a2 = (tmp_list[:, 0] <= start_x + ori_size)
            if (False in a2):
                f["start"]["connect"].append((x + 1) * zero_num + y)
            b1 = tmp_list[:, 1] >= start_y
            zero_num = 10 ** len(str(y - 1))
            if (False in b1):
                f["start"]["connect"].append(x * zero_num + y - 1)
            b2 = tmp_list[:, 1] <= start_y + ori_size
            zero_num = 10 ** len(str(y + 1))
            if (False in b2):
                f["start"]["connect"].append(x * zero_num + y + 1)
            selection = a1 * a2 * b1 * b2
            selection_list=split(selection)
            if(len(selection_list) == 0):
                    continue
            for selection in selection_list:
                tmp_f=copy.deepcopy(f)
                if (False not in selection):
                    tmp_f["pop"] = 1
                tmp_f["geometry"]['coordinates']=tmp_list[selection,:].tolist()
                geojson["features"].append(tmp_f)       
    # with open("./concat.json", 'w') as file_obj:
    #     json.dump(geojson, file_obj)
    return geojson


def findLineIndex(myjs, use, cnn_index):
    line_index = []
    for index in use:
        feature = myjs["features"][index]
        if (feature["pop"] > 0):
            use.remove(index)
            continue
        if (cnn_index in feature["start"]["xy"]):
            line_index.append(index)
    return line_index


def calDis(point1, point2):
    dis = sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return dis


def checkConn(line1, line2, th=5):
    if (len(line1) == 0 or len(line2) == 0):
        return []
    head1 = line1[0]
    bottom1 = line1[-1]
    head2 = line2[0]
    bottom2 = line2[-1]
    if (calDis(head1, head2) < th):
        line1.reverse()
        return line1 + line2
    if (calDis(head1, bottom2) < th):
        line1.reverse()
        line2.reverse()
        return line1 + line2
    if (calDis(bottom1, head2) < th):
        return line1 + line2
    if (calDis(bottom1, bottom2) < th):
        line2.reverse()
        return line1 + line2
    return []


def connect(myjs, use, index):
    new_cnn_list = []
    for cnn_index in myjs["features"][index]["start"]["connect"]:
        line_list = findLineIndex(myjs, use, cnn_index)
        for line_index in line_list:
            tmp = checkConn(myjs["features"][index]["geometry"]["coordinates"],
                            myjs["features"][line_index]["geometry"]["coordinates"])
            if (len(tmp) == 0):
                continue
            myjs["features"][index]["geometry"]["coordinates"] = tmp
            myjs["features"][index]["start"]["xy"] += myjs["features"][line_index]["start"]["xy"]
            new_cnn_list = new_cnn_list + \
                myjs["features"][line_index]["start"]["connect"]
            use.remove(line_index)
            try:
                myjs["features"][index]["start"]["connect"].remove(cnn_index)
            except:
                pass
            myjs["features"][line_index]["pop"] = 2
    myjs["features"][index]["start"]["connect"] += new_cnn_list
    if (len(myjs["features"][index]["start"]["connect"]) != 0):
        myjs["features"][index]["pop"] = 0
    return 0
    # connect(myjs, use, index)


def gen(mask, queue):
    print("start")
    for i in range(32768 // 8192):
        tmp = mask[:, i * 8192:(i + 1) * 8192]
        queue.put(tmp)
        # time.sleep(5)


def main_processing(queue, img_size, size4crop, save_path, geoTransform):

    total_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cur_pointer = 0
    cur_col = 0  # 指向下次开始读取的列
    cur_tol_col = 0  # 当前总的列数
    padSize = 1024
    pad = 100
    oriSize = padSize - (pad * 2)
    cur_list_len = 0  # 每次获取数据得到的线段数量
    next_list_start = 0  # 末尾的下一个线段的索引
    geojson = {}
    use_index = []
    loop = int(img_size / size4crop)
    while (cur_pointer != loop):
        mask = queue.get()  # 等待进入队列 (65536,8192)
        t1 = time.perf_counter()
        json_list = []
        print("cureent index:", cur_pointer)
        total_mask[:, cur_pointer *
                   size4crop:(cur_pointer + 1) * size4crop] = mask
        height = mask.shape[0]
        width = mask.shape[1]
        cur_tol_col += width
        tmp_col = cur_col
        for i in range(0, height+oriSize, oriSize):  # y
            if (i >= height):
                continue
            else:
                tmp_col = cur_col
            for j in range(cur_col, cur_tol_col+oriSize, oriSize):  # x
                if (j < img_size and j+padSize >= img_size):
                    tmp_col = cur_tol_col
                    pass
                elif (j + oriSize+pad >= cur_tol_col):
                    tmp_col = j
                    break
                elif (j + oriSize+pad == cur_tol_col - 1):
                    tmp_col = cur_tol_col
                    break
                tmp = imgPad((j, i), total_mask, (oriSize, oriSize), pad)
                res = CenterlineExtraction(tmp, 0, 0, (j - pad, i - pad))
                tmp_json = {}
                tmp_json["x"] = j
                tmp_json["y"] = i
                tmp_json["data"] = res
                json_list.append(tmp_json)
                # filename="./output/"+"{}_{}_{}.json".format(j,i,pad)
                # with open(filename,'w') as file_obj:
                # json.dump(res,file_obj)
        cur_col = tmp_col
        cur_pointer += 1
        cur_concat = concat(json_list, oriSize)
        cur_list_len = len(cur_concat['features'])
        if (cur_pointer == 1):  # 第一次
            geojson = cur_concat
        else:
            geojson['features'] += cur_concat['features']
        use_index += np.arange(next_list_start,
                               next_list_start + cur_list_len, 1).tolist()
        next_list_start += cur_list_len
        # loop_copy=copy.deepcopy(use_index)
        for index in use_index:
            feature = geojson['features'][index]
            if (feature["pop"] > 0):
                use_index.remove(index)
                continue
            geojson["features"][index]["pop"] = 1
            connect(geojson, use_index, index)
        # print("use_index length:", len(use_index))
        t2 = time.perf_counter()
        print('one column centerline extraction cost time: %f s' % (t2 - t1))

    t3 = time.perf_counter()
    index = 0
    while (True):
        if (geojson["features"][index]["pop"] == 2):
            geojson["features"].pop(index)
        else:
            index += 1
        if(index>=len(geojson["features"])):
            break


    for index, feature in enumerate(geojson['features']):
        tmp_list = geojson['features'][index]["geometry"]["coordinates"]
        tmp_list = simplify_coords_vw(
            tmp_list, 10)
        tmp_list = np.array(tmp_list)
        tmp_list[:, 0] = geoTransform[0] + tmp_list[:, 0] * geoTransform[1]
        tmp_list[:, 1] = geoTransform[3] + tmp_list[:, 1] * geoTransform[5]
        geojson['features'][index]["geometry"]["coordinates"] = tmp_list.tolist()
    filename = save_path + ".json"
    # with open(filename, 'w') as file_obj:
    #     json.dump(geojson, file_obj)
    # ds = gdal.OpenEx(json.dumps(geojson))
    # gdal.VectorTranslate(save_path+'.shp', ds, format='ESRI Shapefile')
    t4 = time.perf_counter()
    print('write in json file cost time: %f s' % (t2 - t1))
    print('geoTransform: %f %f %f %f %f %f' % (
        geoTransform[0], geoTransform[1], geoTransform[2], geoTransform[3], geoTransform[4], geoTransform[5]))


def init_CenterlineExtractor(queue, img_size, size4crop, save_path, geoTransform):
    p = multiprocessing.Process(group=None, target=main_processing, args=(
        queue, img_size, size4crop, save_path, geoTransform))
    p.start()

if __name__ == "__main__":
    # mask=cv2.imread("mask.png",-1)
    # mask = np.load("/nfs/ssdk/output.npz")["arr_0"]
    source='./ZL'
    fileNames=os.listdir(source)
    fileNames=['zj_sx_gd_18_new.png']
    start=time.time()
    for fileName in fileNames:
        filePath=os.path.join(source,fileName)
        outName=fileName.replace('.png','')
        mask=cv2.imread(filePath)[:,:,0]

        print("H:{} W:{}".format(mask.shape[0],mask.shape[1]))
        # mask=np.zeros((65536,65536),dtype=np.uint8)
        queue = multiprocessing.Queue(4)
        p1 = multiprocessing.Process(group=None, target=gen, args=(mask, queue,))
        geoTrans=(121.640625,0.000005,0.000000,29.179688,0.000000,-0.000005)
        geoTrans2=(0,1,0,0,0,-1)
        p2 = multiprocessing.Process(
            group=None, target=main_processing, args=(queue,16384,8192,outName,geoTrans2,))
            #(queue, img_size, size4crop, save_path, geoTransform)
        p1.start()
        p2.start()
        print(queue.qsize())
        # time.sleep(0.1)
        print("Sub-process all done.")
    print("all time:",time.time()-start)
        # CenterlineExtractionProcess(mask,"./")
