import numpy as np
import json
import requests
import cv2
import time
import threading

class Camera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        #self.video1 = cv2.VideoCapture(1)
        #self.video = cv2.VideoCapture('output.mp4')
        #self.video = cv2.VideoCapture('test.jpg')

    def __del__(self):
        self.video.release()

    def get_frame(self):
            ret, frame = self.video.read()
            #ret2, frame2 = self.video1.read()
            r, jpg = cv2.imencode('.jpg', frame)
            return frame
        #return frame

    def cam_open(self):
        opened = self.video.isOpened()
        return opened
'''

class Camera(object):
    thread = {}
    frame = {}
    last_access = {}
    event = {}
    
    def __init__(self, camera_type=None, device=None):
        self.unique_name = "{cam}_{dev}".format(cam=camera_type, dev=device)
        Camera.event[self.unique_name] = CameraEvent()
        if self.unique_name not in Camera.thread:
            Camera.thread[self.unique_name] = None
        if Camera.thread[self.unique_name] is None:
            Camera.last_access[self.unique_name] = time.time()

            Camera.thread[self.unique_name] = threading.Thread(targets=self._thread,
                                                               args=(self.unique_name,))
            Camera.thread[self.unique_name].start()

            while self.get_frame() is None:
                time.sleep(0)


    def get_frame(self):
        """Return the current camera frame."""
        Camera.last_access[self.unique_name] = time.time()

        # wait for a signal from the camera thread
        Camera.event[self.unique_name].wait()
        Camera.event[self.unique_name].clear()

        return Camera.frame[self.unique_name]

    def frames():
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    def _thread(cls, unique_name):
        """Camera background thread."""
        print('Starting camera thread')
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            Camera.frame[unique_name] = frame
            Camera.event[unique_name].set()  # send signal to clients
            time.sleep(0)

            # if there hasn't been any clients asking for frames in
            # the last 5 seconds then stop the thread
            if time.time() - Camera.last_access[unique_name] > 5:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity')
                break
        Camera.thread[unique_name] = None

'''     
