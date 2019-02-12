import numpy as np
import cv2
import math
import pickle

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

frameRate = cap.get(5) #frame rate
x=1
decimg = ""

while(cap.isOpened()):
    frameId = cap.get(10) 
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        filename = './test' +  str(int(x)) + ".png";x+=1
        cv2.imwrite(filename, frame)
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        results, encimg = cv2.imencode('.png', frame, encode_param)

        decimg = cv2.imdecode(encimg, 1)

        cv2.imshow('frame',decimg)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
