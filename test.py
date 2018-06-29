import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_yaml
from skimage.feature import canny
# import matplotlib.pyplot as plt
import sys
import os
import cv2

RUN_ID = "5"

def load_images(dirname):
    imlist =[]
    for fname in os.listdir(dirname):
        im = np.array(cv2.imread(dirname+fname))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im[10:-10, 90:-90]
        im = (im - np.average(im))/np.std(im)
        im_canny = np.resize(canny(im_gray), (im.shape[0], im.shape[1], 1))
        im = np.concatenate((im, im_canny), axis=2)
        imlist.append(im)

    imlist = np.array(imlist)
    return imlist

model = model_from_yaml(open("model/" + RUN_ID + ".yaml").read())
model.load_weights( "model/" + RUN_ID + "_weight.h5")
imlist_test = load_images("data/test/")
x_test = np.concatenate([imlist_test],axis=0)
#     ０はOK、１はNG
y_test = np.array([0])

    #OK:0,NG:1を返す
y_pred = np.round(model.predict(x_test, batch_size = 48, verbose=1))
y_pred = y_pred.flatten()

if y_pred[0] > 1:
        print('NG!アーム動かします！')
if y_pred[0] < 1:
        print('OK!アーム動かしません!')

