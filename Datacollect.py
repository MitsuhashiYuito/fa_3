import numpy as np
import cv2
import time
import subprocess

from skimage.feature import canny
import sys
import os
from glob import glob
import platform
from pydobot import Dobot


def load():
    if platform.system() == "Darwin" :
        available_ports = glob('/dev/cu*usb*')  # mask for OSX Dobot port
    else:
        pass

available_ports = glob('/dev/ttyUSB0')  # mask for Raspi Dobot port
if len(available_ports) == 0:
    print('no port found for Dobot Magician')
    exit(1)

def load_images(dirname):
    imlist =[]
    for fname in os.listdir(dirname):
        im = np.array(cv2.imread(dirname+fname))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = (im - np.average(im))/np.std(im)
        im = im[10:-10, 90:-90]
        im_canny = np.resize(canny(im_gray), (im.shape[0], im.shape[1], 1))
        im = np.concatenate((im, im_canny), axis=2)
        imlist.append(im)

    imlist = np.array(imlist)
    return imlist
device = Dobot(port=available_ports[0])
device.move_conveyor_belt(0.6, direction=1)
time.sleep(.1)
device.close()

CROP_W, CROP_H = 150,200
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(frame.shape)
imlist_test = load_images("data/test/")
x_test = np.concatenate([imlist_test],axis=0)

th = 30
ret, bg = cap.read()
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

skip = 0

while(True):
    skip -= 1
    ret, frame = cap.read()
    if skip > 0:
      continue
    X,Y,Z = -40, 258.2, 4.8
    a,b,c = 242, -10, 15
    W, H, _ = frame.shape
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgbgmask = cv2.absdiff(frame_gray, bg_gray)
    fgbgmask[fgbgmask < th] = 0
    fgbgmask[fgbgmask > th] = 255
    #crop
    ROI = fgbgmask[CROP_H:-CROP_H, CROP_W:-CROP_W]    
    NUM_WHITE = cv2.countNonZero(ROI)
    print(NUM_WHITE)    

    mask = cv2.inRange(frame, (0,0,0), (200,200,200))
    frame_canny = cv2.Canny(mask, threshold1=150, threshold2=300)

    # crop RoI
    roi = frame_canny[CROP_H:-CROP_H, CROP_W:-CROP_W]

    num_white = cv2.countNonZero(roi)
    is_exist = True if NUM_WHITE > 10000 else False
    if is_exist:
        # find moments
        print(str(NUM_WHITE) + "exist")
        mu = cv2.moments(roi, False)
        x, y = int(mu["m10"]/mu["m00"])+CROP_W, int(mu["m01"]/mu["m00"])+CROP_H

    cv2.imshow('demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        device = Dobot(port=available_ports[0])
        device.stop_conveyor_belt()
        device.close()
        break
    if is_exist:
        now = time.time()
        device = Dobot(port=available_ports[0])
        device.stop_conveyor_belt()
        
        cv2.imwrite('data2/NG2/' + str(now) +  '.png',frame)
        device.move_conveyor_belt(0.6, direction=1)
        time.sleep(.1)
        device.close() 
        skip = 40
cap.release()
cv2.destroyAllWindows()
