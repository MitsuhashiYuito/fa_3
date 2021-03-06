import numpy as np
import cv2
import time
import subprocess
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_yaml
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



model = model_from_yaml(open("model/9.yaml").read())
model.load_weights( "model/9_weight.h5")


CROP_W, CROP_H = 150,200
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(frame.shape)
imlist_test = load_images("data/test/")
x_test = np.concatenate([imlist_test],axis=0)
y_pred = np.round(model.predict(x_test, batch_size = 48, verbose=1))

th = 30
ret, bg = cap.read()
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

skip = 0

while(True):
    skip -= 1
    ret, frame = cap.read()
    if skip > 0:
      continue
    X,Y,Z = -100, 258.2, 2.8
    a,b,c = 242, 60, 25
    W, H, _ = frame.shape
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgbgmask = cv2.absdiff(frame_gray, bg_gray)
    fgbgmask[fgbgmask < th] = 0
    fgbgmask[fgbgmask > th] = 255
    showfgbgmask = cv2.rectangle(fgbgmask,(130,180),(510,300),(255,0,0),3)
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
 #       frame = cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)

#    output_image = np.zeros((W*2, H*2, 3))
#    output_image[:W, :H] = frame/255
#    output_image[W:, :H] = np.stack((mask, mask, mask), axis=-1)
#    output_image[:W, H:] = np.stack((frame_canny, frame_canny, frame_canny), axis=-1)
#    output_image[W+CROP_H:-CROP_H, H+CROP_W:-CROP_W] = np.stack((roi, roi, roi), axis=-1)

    cv2.imshow('demo', showfgbgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        device = Dobot(port=available_ports[0])
        device.stop_conveyor_belt()
        device.close()
        break
    if is_exist:

        device = Dobot(port=available_ports[0])
        device.stop_conveyor_belt()
        cv2.imwrite('data/test/' + "1" +  '.png',frame)
        imlist_test = load_images("data/test/")
        x_test = np.concatenate([imlist_test],axis=0)
        y_test = np.array([0])
    #OK:0,NG:1を返す
        y_pred = np.round(model.predict(x_test, batch_size = 48, verbose=1))
        y_pred = y_pred.flatten()
        skip = 40
        if y_pred[0] == 1:
            print('NG!!!!!')
            time.sleep(.5)
            device.speed(50)
            device.go(X,Y,Z+20)
            device.speed(50)
            device.go(X,Y,Z)
            time.sleep(.5)

            device.suck(True)
            time.sleep(.5)

            device.speed(30)
            device.go(X,Y,Z+20)
            device.speed(30)
            device.go(a,b,+20)
            device.speed(30)
            device.go(a,b,c-10)
            time.sleep(.5)

            device.suck(False)

            device.move_conveyor_belt(0.6, direction=1)
            time.sleep(.1)
	
            device.close()


            
        if y_pred[0] == 0:
            print('OK!!!!!!')
            device.move_conveyor_belt(0.6, direction=1)
            time.sleep(.1)
            device.close()
#          print('now capture', now)
        
        
cap.release()
cv2.destroyAllWindows()
