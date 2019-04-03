from app import db
from app.models import Frame
import numpy as np
import cv2
import math
import pickle
import matplotlib.pyplot as plt
import importlib
import os, glob

cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            filename = 'test.jpg';
            cv2.imwrite(filename, frame)
            break
    else:
        break
u = Frame(frame_fname = filename, video_fname = 'output.mp4')
db.session.add(u)
db.session.commit()

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
