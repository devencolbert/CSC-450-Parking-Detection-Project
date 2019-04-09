import numpy as np
import cv2
import math
import yaml
import matplotlib.pyplot as plt
import importlib
import os, glob
import json
import base64
import requests

class Camera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        #r, encimg = cv2.imencode('.jpg', frame)
        return frame

    def get_sec(self):
        time = self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return time

    def get_frame_pos(self):
        frame_pos = self.video.get(cv2.CAP_PROP_POS_FRAMES)
        return frame_pos
'''
cam_id = 0
cap = cv2.VideoCapture(cam_id)
time.sleep(2.0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output1.mp4',fourcc, 20.0, (640,480))

    

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            filename = './test' +  str(int(1)) + ".jpg";
            cv2.imwrite(filename, frame)
            str = base64.b64encode(frame)
            #print(str)
            #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
            #encimg = cv2.imencode('.jpg',frame,encode_param)
            #cam_frame = json.dumps(encimg)
            #requests.post('http://httpbin.org/post',data=cam_frame)
            break
    else:
        break

info = {'cam_id': 0, 'camera': []}
info['cam_id'] = cam_id
info['camera'] = ['webcam']
data = []
data.append(info)
with open('camera_id.yml', 'a') as file:
    yaml.dump(data, file)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
'''
