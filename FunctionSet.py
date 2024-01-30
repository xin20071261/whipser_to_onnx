# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:47:52 2022

@author: admin
"""

import numpy as np
import os
import json
import path
from pathlib import Path
import yaml
import csv

def save_data(name,data,mPath):
    isExists = os.path.exists(mPath)
    if not isExists:
        os.makedirs(mPath)
    mPath = os.path.join(mPath,name)
    print(mPath)
    np.savez(mPath,dt={name:data})
    
def load_data(name,mPath = 'data'):
    mPath = os.path.dirname(os.path.abspath(__file__))+"/"+mPath
    mPath = os.path.join(mPath, name)
    data = np.load(mPath+".npz",allow_pickle = True)
    data = data["dt"][()]
    data = data[name]
    return data

def load_txt(mPath):
    data = []
    with open(mPath,'r') as f:
        for line in f.readlines():
            data.append(line)
    return data

def save_txt(mPath,data):
    with open(mPath,'w') as f:
        for dt in data:
            f.write(str(dt))
            f.write("\n")

def save_txt_p(mPath,data):
    with open(mPath,'a') as f:
        f.write(str(data))
        f.write("\n")    

#mPath是完整路径
def load_json(mPath):
    with open(mPath,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    return json_data
#读取list方式的json
def load_json_list(mPath):
    data = []
    with open(mPath,'r',encoding='utf8')as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data
#读取csv
def load_csv(mPath):
    data = ""
    with open(mPath,'r',encoding = 'utf8') as f:
        csv_data = f.read()
        #for row in csv_data:
        data = csv_data
        data = data.split("\n")
    for i in range(len(data)):
        data[i] = data[i].split(";")
    return data

#读取文件夹下的文件名称,子目录等，并拼接成完整路径
def file_name(file_dir): 
    rootSet = []
    fileSet = []
    dirSet = []
    roadSet = []
    for root, dirs, files in os.walk(file_dir):
        rootSet.append(root)
        dirSet.append(root)
        fileSet.append(files)
    for i in range(len(dirSet)):
        for file in fileSet[i]:
            roadSet.append(os.path.join(dirSet[i], file))
    #return rootSet,dirSet,fileSet,roadSet
    return roadSet

#计算直线和y轴的焦点，用于车道线排序
def getYNode(Point_1,Point_2):
    Point_1[1] = 1440-Point_1[1]#x为图像的行数，坐标中心由右上转右下
    Point_2[1] = 1440-Point_2[1]
    y_0 = Point_2[0]*Point_1[1] - Point_1[0]*Point_2[1]
    
    b = Point_1[1] - Point_2[1]
    if b != 0:
        y_0 /= b
    else:
        y_0 = -10000 #和x轴没有交点，直接剔除
    return y_0
        
#创建文件夹
def makeMyDirs(mmPath,childmPath):
    if(os.mPath.exists(mmPath)):
        myPath = os.path.join(mmPath, childmPath)
    else:
        print("主目录不存在！！！")
        return
    if(os.path.exists(myPath) == False):
        os.makedirs(myPath)
    return myPath
    
#读取yaml文件
def load_yaml(mPath):
    f = open(mPath,"r",encoding = "utf8")
    cfg = f.read()
    dt = yaml.load(cfg,Loader = yaml.FullLoader)
    #Loader = yaml.FullLoader
    return dt
