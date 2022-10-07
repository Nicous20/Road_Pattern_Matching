import json
from multiprocessing.dummy import Pool
import sys
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from tqdm import tqdm
sys.setrecursionlimit(2000000)
k=0
step=0
def judge(mask,pos):
    H,W=mask.shape
    posL=[]
    y,x=pos
    for i in range(-1,2):
        for j in range(-1,2):
            if(y+i<0):continue
            if(x+j<0):continue
            if(y+i>=H):continue
            if(x+j>=W):continue
            if(mask[y+i,x+j]>100):
                posL.append((y+i,x+j))
    return posL
def find(mask,img,pos,length,intersections):
    global k,step
    length=length
    y,x=pos
    if(mask[y,x]==2): return 1024
    while(True):
        mask[y,x]=1 #走过的路线
        step+=1
        if(step%100==0):
            k+=1
            # cv2.putText(img, str(k), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)
        posL=judge(mask,(y,x))
        crossNum=len(posL)
        if(crossNum==1):
            y,x=posL[0]
            length+=1
            continue
        if(crossNum>1):
            lenList=[]
            for pos in posL:
                lenList.append(find(mask,img,pos,1,intersections))
            lenList=sorted(lenList,reverse=True)
            # k+=1
            # cv2.putText(img, str(lenList), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)
            if(lenList[1]>200):
                intersections.append((y,x))
                mask[y,x]=2 #标记节点
                return 1024
            return length+lenList[0]
        mask[y,x]=2
        return length
def singleThread(filename,idx):
    intersections=[]
    ROOT='skeleton'
    filepath=join(ROOT,filename)
    img=cv2.imread(filepath,0)
    mask=np.array(img).copy()
    H,W=mask.shape
    for i in range(H):
        for j in range(W):
            if(mask[i,j]>100):
                find(mask,img,(i,j),1,intersections)
    for pos in intersections:
        y,x=pos
        cv2.circle(img,(x,y),40,(20,255,255),-1)
    intersections=sorted(intersections,key=lambda tup:tup[0])
    # print(intersections)
    newInter=[]
    for id,sel in enumerate(intersections):
        if(id==0):
            newInter.append(sel)
            continue
        yOld,xOld=intersections[id-1]
        if(abs(sel[0]-yOld)>3 or abs(sel[1]-xOld)>3):
            newInter.append(sel)
    print("Thread: ",idx," is done!")
    return {'path':filepath,'inter':newInter}
    # print(newInter)
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(img)
    # plt.show()
from multiprocessing import Process,Pool
def test():
    result=[]
    ROOT='skeleton'
    filenames=os.listdir(ROOT)
    p=Pool(15)
    for idx,filename in enumerate(filenames):
        result.append(p.apply_async(singleThread, args=(filename,idx,)))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    res=[r.get() for r in result]
        # result.append({'path':filepath,'inter':intersections})
    with open('./intersection2.json','w') as f:
        json.dump(res,f)
if __name__ == '__main__':
    test()