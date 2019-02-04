import numpy as np
import cv2
import math

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

frameRate = cap.get(5) #frame rate
x=1

while(cap.isOpened()):
    frameId = cap.get(1) 
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        filename = './test' +  str(int(x)) + ".jpg";x+=1
        cv2.imwrite(filename, frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
