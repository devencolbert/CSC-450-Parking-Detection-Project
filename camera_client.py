import numpy as np
import cv2
import math
import pickle
import matplotlib.pyplot as plt
import importlib
import os, glob

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

frameRate = cap.get(5) #frame rate
x=1
i=0
decimg = []
while(cap.isOpened()):
    frameId = cap.get(10) 
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        filename = './test' +  str(int(x)) + ".png";x+=1
        cv2.imwrite(filename, frame)
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        results, encimg = cv2.imencode('.png', frame, encode_param)
        #images = [plt.imread(path) for path in glob.glob('test/*.jpg')]
        decimg = cv2.imdecode(encimg, 1)
        #i += 1

        cv2.imshow('frame',frame)
        cv2.imshow('Decoded image',decimg)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        break


# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
