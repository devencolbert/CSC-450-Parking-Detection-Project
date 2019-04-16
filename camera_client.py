import numpy as np
import json
import requests
import cv2

class Camera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        #self.video = cv2.VideoCapture('output.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        r, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
        #return frame

    def cam_open(self):
        opened = self.video.isOpened()
        return opened
