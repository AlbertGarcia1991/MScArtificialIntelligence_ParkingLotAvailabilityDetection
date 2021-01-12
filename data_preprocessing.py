########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                    INM705 - Garcia Plaza, Albert / Bohkary, Syed                                     #
########################################################################################################################

import numpy as np
import os, shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


# Directories and variables initialize
cPath = os.getcwd() + os.sep
filesOnCwd = os.listdir(cPath)

# Original images size
widthOriginal = 2592
heightOriginal = 1944

# Size as CNN input
width = 1000
height = 750
cameraControl = 5


# Read bbox .csv files
bboxFiles = sorted([csvFile for csvFile in filesOnCwd if csvFile.endswith('.csv')])
bboxes = []
for bboxFile in bboxFiles:
    bboxes_ = np.genfromtxt(cPath+bboxFile, delimiter=',', skip_header=True)
    bboxes.append(bboxes_)
bboxesAdim = []
for bbox in bboxes:
    bboxesAdim_ = []
    for i in range(len(bbox)):
        bboxesAdim__ = []
        bboxesAdim__.append(int(bbox[i][0]))
        bboxesAdim__.append((bbox[i][1] + (bbox[i][3] / 2 )) / widthOriginal)
        bboxesAdim__.append((bbox[i][2] - (bbox[i][4] / 2 )) / heightOriginal)
        bboxesAdim__.append(bbox[i][3] / widthOriginal)
        bboxesAdim__.append(bbox[i][4] / heightOriginal)
        bboxesAdim__.append(0)
        bboxesAdim_.append(bboxesAdim__)
    bboxesAdim.append(bboxesAdim_)


# Read lot availability file for all images
with open(cPath + 'all.txt', 'r') as f:
    lines = f.readlines()
    lines = [line[:-1] for line in lines]
    lines = [line.split('/')[-1] for line in lines]
    lines = [line[2:] for line in lines]
states = []
for line in lines:
    state_ = []
    state_.append(int(line[-1]))
    line_ = line.split('C')[0][:-1]
    line_ = line_.split('.')[0] + line_.split('.')[1]
    state_.append(line_)
    line_ = int(line.split('C')[1][3:-6])
    state_.append(line_)
    states.append(state_)
states = [state for state in states if state[0] == 1]


# Create txt bbxes files
def create_bbox_files_available(camera, filename):
    index = camera - 1
    filename = filename.split('.')[0] + '.txt'
    file = cPath + 'Full_Dataset' + os.sep + filename
    newLines = ""
    for i in range(len(bboxesAdim[index])):
        available = True

        for state in states:
            if state[1] == filename.split('.')[0]:
                if state[-1] == int(bboxesAdim[index][i][0]):
                    available = False

        if available:
            newLines += str(0) + " "
        else:
            newLines += str(1) + " "
        newLines += str(bboxesAdim[index][i][1]) + " "
        newLines += str(bboxesAdim[index][i][2]) + " "
        newLines += str(bboxesAdim[index][i][3]) + " "
        newLines += str(bboxesAdim[index][i][4]) + "\n"
    with open(file, 'w+') as f:
        f.write(newLines)


# Move all images from original folders to 'Full_Dataset' directory
def copy_from_original_path():
    weathers = ['OVERCAST', 'RAINY', 'SUNNY']
    cameras = ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']
    dst = cPath + 'Full_Dataset' + os.sep
    for weather in weathers:
        dates = os.listdir(cPath + 'FULL_IMAGE_1000x750' + os.sep + weather + os.sep)
        for date in dates:
            for camera in cameras:
                path = cPath + 'FULL_IMAGE_1000x750' + os.sep + weather + os.sep + date + os.sep + camera + os.sep
                imgs = os.listdir(path)
                for img in imgs:
                    src = path + img
                    shutil.copy(src, dst)
                    create_bbox_files_available(int(camera[-1]), img)


# Split dataset into block1, block2 and test block
def split_dataset():
    files = os.listdir(cPath + 'Full_Dataset' + os.sep)
    imgs = sorted([file for file in files if file.endswith('.jpg')])
    txts = sorted([file for file in files if file.endswith('.txt')])
    index = np.arange(len(imgs))
    random.shuffle(index)
    cutoff = int(len(imgs) * 0.4)
    counter = 0
    for i in range(len(index)):
        srcImg = cPath + 'Full_Dataset' + os.sep + imgs[index[i]]
        srcTxt = cPath + 'Full_Dataset' + os.sep + txts[index[i]]
        if counter < cutoff:
            shutil.copy(srcImg, cPath + 'block1' + os.sep)
            shutil.copy(srcTxt, cPath + 'block1' + os.sep)
        elif cutoff < counter < 2 * cutoff:
            shutil.copy(srcImg, cPath + 'block2' + os.sep)
            shutil.copy(srcTxt, cPath + 'block2' + os.sep)
        else:
            shutil.copy(srcImg, cPath + 'test' + os.sep)
            shutil.copy(srcTxt, cPath + 'test' + os.sep)
        counter += 1


# Create .txt data files to be read by YOLOv3s
def create_path_files():
    blocks = ['block1', 'block2']
    for block in blocks:
        files = os.listdir(cPath + block + os.sep)
        txts = [file for file in files if file.endswith('.txt')]
        newLine = ""
        for txt in txts:
            newLine += "%" + txt + "\n"
        filename = block + ".txt"
        with open(filename, 'w+') as f:
            f.write(newLine)
    files = os.listdir(cPath + 'test' + os.sep)
    txts = [file for file in files if file.endswith('.txt')]
    newLine = ""
    for txt in txts:
        newLine += "%" + txt + "\n"
    with open("test.txt", 'w+') as f:
        f.write(newLine)

create_path_files()
