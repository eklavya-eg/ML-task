# name: Eklavya Gupta
# university: University of Petroleum and Energy Studies
# course: B.Tech CSE AI ML   # currently in 5th semester
# sapid: 500093960
# university mail id: 500093960@stu.upes.ac.in
# personal mail id: emessage.eg@gmail.com


import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random
import PIL
from PIL import Image

# data-development utils
# rotation
def augmentedRotation(imageDir:str, savingDir:str, angles:list, scalingFactor:int = 1):
    try:
        os.mkdir(path = savingDir)
        print(f'Directory Created: {savingDir}')
    except FileExistsError:
        pass
    for angle in angles:
        if angle == 90 or angle == 270:
            try:
                imgs = os.listdir(imageDir)
                for i, image in enumerate(imgs):
                    img = cv2.imread(os.path.join(imageDir, image))
                    rotation = cv2.getRotationMatrix2D(center = (img.shape[1]//2, img.shape[0]//2), angle = angle, scale = 1)
                    img = cv2.warpAffine(img, rotation, (img.shape[0], img.shape[1]))
                    cv2.imwrite(os.path.join(savingDir, f'augmented-rotation-{imgs[i]}-{i}.jpg'), img = img)
                print(f'done angle {angle}')
            except FileNotFoundError:
                print(f'Image Directory not found')
        elif angle == 180 or angle == 360:
            try:
                imgs = os.listdir(imageDir)
                for i, image in enumerate(imgs):
                    img = cv2.imread(os.path.join(imageDir, image))
                    rotation = cv2.getRotationMatrix2D(center = (img.shape[1]//2, img.shape[0]//2), angle = angle, scale = 1)
                    img = cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
                    cv2.imwrite(os.path.join(savingDir, f'augmented-rotation-{imgs[i]}-{i}.jpg'), img = img)
                print(f'done angle {angle}')
            except FileNotFoundError:
                print(f'Image Directory not found')
        else:
            try:
                imgs = os.listdir(imageDir)
                for i, image in enumerate(os.path.join(imageDir, imgs)):
                    img = cv2.imread(image)
                    rotation = cv2.getRotationMatrix2D(center = (img.shape[1]//2, img.shape[0]//2), angle = angle, scale = 1)
                    img = cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
                    cv2.imwrite(os.path.join(savingDir, f'augmented-rotation-{imgs[i]}-{i}.jpg'), img = img)
                print(f'done angle {angle}')
            except FileNotFoundError:
                print(f'Image Directory not found')



# shearing  
def augmentedShearing(imageDir:str, savingDir:str, shearFactor:[tuple], scalingFactor:int = 1):
    try:
        os.mkdir(path = savingDir)
        print(f'Directory Created: {savingDir}')
    except FileExistsError:
        pass
    for shearX, shearY in shearFactor:
        try:
            imgs = os.listdir(imageDir)
            for i, image in enumerate(imgs):
                img = cv2.imread(os.path.join(imageDir, image))
                shear = np.float32([[1, shearX, 0], [shearY, 1, 0]])
                img = cv2.warpAffine(img, shear, (img.shape[1], img.shape[0]))
                cv2.imwrite(os.path.join(savingDir, f'augmented-shear-{imgs[i]}-{i}.jpg'), img = img)
            print(f'done shear factor{(shearX, shearY)}')
        except FileNotFoundError:
            print(f'Image Directory not found')



# mirroring
def augmentedMirroring(imageDir:str, savingDir:str, flipCode = 1, ):
    try:
        os.mkdir(path = savingDir)
        print(f'Directory Created: {savingDir}')
    except FileExistsError:
        pass
    try:
        imgs = os.listdir(imageDir)
        for i, image in enumerate(imgs):
            img = cv2.imread(os.path.join(imageDir, image))
            img = cv2.flip(img, flipCode=flipCode)
            cv2.imwrite(os.path.join(savingDir, f'augmented-mirror-{imgs[i]}-{i}.jpg'), img = img)
        print(f'done')
    except FileNotFoundError:
        print(f'Image Directory not found')



# resizing
def resizingImages(imageDir:str, savingDir:str, size:tuple):
    try:
        os.mkdir(path = savingDir)
        print(f'Directory Created: {savingDir}')
    except FileExistsError:
        pass
    try:
        imgsName = os.listdir(imageDir)
        imgs = [(cv2.imwrite(filename = os.path.join(savingDir, imgsName[index]), img = cv2.resize(cv2.imread(os.path.join(imageDir, image)), size, interpolation = cv2.INTER_LINEAR))) for index, image in enumerate(os.listdir(imageDir), 0)]
        print(f'Images Directory: {savingDir}')
    except FileNotFoundError:
        print('Image directory not found')



# getting max and min sizes
def maxHeightWidth(dir:str):
    imgs = os.listdir(dir)
    imgsH = [plt.imread(os.path.join(dir, image)).shape[0] for image in imgs]
    imgsW = [plt.imread(os.path.join(dir, image)).shape[0] for image in imgs]
    print(f'maximum height: {max(imgsH)}')
    print(f'maximum width: {max(imgsW)}')
    return (max(imgsH), max(imgsW))

def minHeightWidth(dir:str):
    imgs = os.listdir(dir)
    imgsH = [plt.imread(os.path.join(dir, image)).shape[0] for image in imgs]
    imgsW = [plt.imread(os.path.join(dir, image)).shape[0] for image in imgs]
    print(f'maximum height: {min(imgsH)}')
    print(f'maximum width: {min(imgsW)}')
    return (min(imgsH), min(imgsW))



# image data generartion
def DirToCsv(dir:str, filename:str, shuffle:bool = False):
    labels = os.listdir(dir)
    dataList = []
    for label in labels:
        labelDir = os.listdir(os.path.join(dir, label))
        for name in labelDir:
            dataList.append((os.path.join(dir, label, name), label))
    if shuffle:
        random.shuffle(dataList)
    with open(filename, 'w') as file:
        file.write(f'location,label\n')
        for imgLoc, label in dataList:
            file.write(f'{imgLoc},{label}\n')