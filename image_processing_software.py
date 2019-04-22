import yaml
import numpy as np
import cv2
import os
import time
import socket
import json
import base64
from PIL import Image
from flask import request
import requests
from io import BytesIO
import urllib.request
#from camera_client import Camera
#import detect_cars.new_segment


def run_classifier(img, id, car_cascade):
    cars = car_cascade.detectMultiScale(img, 1.1, 1)
    if cars == ():
        return False
    else:
        return True


def parking_datasets(fn_yaml):
    with open(fn_yaml, 'r') as stream:
        parking_data = yaml.load(stream)
    return parking_data

def yaml_loader(file_path):
    with open(file_path, "r") as file_descr:
        data = yaml.load(file_descr)
        return data

def yaml_dump(file_path, data):
    with open(file_path, "a") as file_descr:
        yaml.dump(data, file_descr)
        
def get_parking_info(parking_data):    
    parking_contours = []
    parking_bounding_rects = []
    parking_mask = []
    parking_data_motion = []
    if parking_data != None:
        for park in parking_data:
            points = np.array(park['points'])
            rect = cv2.boundingRect(points)
            points_shifted = points.copy()
            points_shifted[:,0] = points[:,0] - rect[0] # shift contour to region of interest
            points_shifted[:,1] = points[:,1] - rect[1]
            parking_contours.append(points)
            parking_bounding_rects.append(rect)
            mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                        color=255, thickness=-1, lineType=cv2.LINE_8)
            mask = mask==255
            parking_mask.append(mask)
    return parking_contours, parking_bounding_rects, parking_mask, parking_data_motion;

def get_car_info(data):
    car_rects = []
    car_bounding_rects = []
    if data != None:
        for coor in data:
            coors = np.array(coor['coors'])
            rect = cv2.boundingRect(coors)
            car_bounding_rects.append(rect)
    return car_bounding_rects

def set_erode_kernel():
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    return kernel_erode

def set_dilate_kernel():
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(5,19))
    return kernel_dilate

def status(parking_data):
    if parking_data != None:
        parking_status = [False]*len(parking_data)
        parking_buffer = [None]*len(parking_data)
    return parking_status, parking_buffer;

def parking_info(spot):
    info = {'id':0, 'availability': ''}
    data = []
    file_path = "parking_info.yml"
    if data == []:
            
        if yaml_loader(file_path) != None:
            if len(yaml_loader(file_path)) == 10:
                os.remove('parking_info.yml')
                file_path = "parking_info.yml"
                new_data = 0
            else:
                new_data = len(yaml_loader(file_path))
        else:
            new_data = 0
    else:
       if yaml_loader(file_path) != None:
           new_data = len(data) + len(yaml_loader(file_path))
       else:
           new_data = len(data)

    info['id'] = new_data + 1
    info['availability'] = spot
    data.append(info)
    if data != []:
        yaml_dump(file_path, data)

def detect_cars():
    #from .detect_cars.new_segment import *
    fn = "car_coor.yml"
    with open(fn, 'r') as stream:
        car_coor = yaml.load(stream)
    return car_coor

def print_parkIDs(park, points, line_img, car_cascade,
                  parking_bounding_rects, grayImg, parking_data, parking_status, vpl):

    spots_change = 0
    total_spots = len(parking_data)
    for ind, park in enumerate(parking_data):
        points = np.array(park['points'])
        if parking_status[ind]:
            color = (0,255,0)
            spots_change += 1
            spot = 'Available'
            rect = parking_bounding_rects[ind]
            roi_gray_ov = grayImg[rect[1]:(rect[1] + rect[3]),
                           rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
            #res = run_classifier(roi_gray_ov, ind, car_cascade)
            #if res:
            #    parking_data_motion.append(parking_data[ind])
                
            #    color = (0,0,255)
        else:
            color = (0,0,255)
            spot = 'Unavailable'
        
        cv2.drawContours(line_img, [points], contourIdx=-1,
                             color=color, thickness=2, lineType=cv2.LINE_8)
        cv2.drawContours(vpl, [points], contourIdx=-1,
                             color=color, thickness=2, lineType=cv2.LINE_8)
        
        moments = cv2.moments(points)
        centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
        # putting numbers on marked regions
        cv2.putText(line_img, str(park['id']), (centroid[0]+1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(line_img, str(park['id']), (centroid[0]-1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(line_img, str(park['id']), (centroid[0]+1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(line_img, str(park['id']), (centroid[0]-1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(line_img, str(park['id']), centroid, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.putText(vpl, str(park['id']), (centroid[0]+1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(vpl, str(park['id']), (centroid[0]-1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(vpl, str(park['id']), (centroid[0]+1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(vpl, str(park['id']), (centroid[0]-1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(vpl, str(park['id']), centroid, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        #parking_info(spot)
        
    

    #Display number of available parking spaces on video for each frame change.
    spots_on_frame = "%d/%d" % (spots_change, total_spots)
    cv2.putText(line_img, spots_on_frame  + ' spaces are available', (6,61), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(line_img, spots_on_frame  + ' spaces are available', (4,59), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(line_img, spots_on_frame  + ' spaces are available', (6,59), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(line_img, spots_on_frame  + ' spaces are available', (4,61), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(line_img, spots_on_frame  + ' spaces are available', (5,60), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (0,0,0), 2, cv2.LINE_AA)

        


def detection(parking_bounding_rects, parking_data, parking_status,
              parking_buffer, grayImg, start, parking_mask, line_img, car_cascade, vpl):

    spots_change = 0
    '''coors = []
    car_rect = []
    car_bounding_rects = get_car_info(car_coor)
    for ind, coor in enumerate(car_coor):
        coors = np.array(coor['coors'])
        car_rect = car_bounding_rects[ind]'''
    # detecting cars and vacant spaces
    for ind, park in enumerate(parking_data):
        points = np.array(park['points'])
        rect = parking_bounding_rects[ind]

        '''if points[2].all() > coors[1].all() or points[0].all() > coors[0].all():
            #print(points)
            overlap = False
        else:
            overlap = True'''
        roi_gray = grayImg[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])] # crop roi for faster calcluation

        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
        
        points[:,0] = points[:,0] - rect[0] # shift contour to roi
        points[:,1] = points[:,1] - rect[1]
        delta = np.mean(np.abs(laplacian * parking_mask[ind]))
        
        pos = time.time()
        
        status = delta < 2.2
        # If detected a change in parking status, save the current time
        if status != parking_status[ind] and parking_buffer[ind]==None:
            parking_buffer[ind] = pos
            
        # If status is still different than the one saved and counter is open
        elif status != parking_status[ind] and parking_buffer[ind]!=None:
            if pos - parking_buffer[ind] > 1:
                parking_status[ind] = status
                parking_buffer[ind] = None
        # If status is still same and counter is open
        elif status == parking_status[ind] and parking_buffer[ind]!=None:
            parking_buffer[ind] = None

# changing the color on the basis on status change occured in the above section and putting numbers on areas
    
        print_parkIDs(park, points, line_img, car_cascade,
                      parking_bounding_rects, grayImg, parking_data, parking_status, vpl)

def requestImage():
    url = "http://localhost:8000/video_feed"
    '''pic = requests.get(url)
    if pic.status_code == 200:
        p = pic.content
        p = base64.b64decode(p)
        cv2.imshow('p', p)
    #return p
        with open('image.jpg', 'w') as f:
            f.write(p)'''
    r = requests.get(url)
    print(r.content)
    with open('image.jpg', 'wb') as f:
        f.write(r.content)
    #return r
        
    
def main():

    
    #cap = Camera()
    frame_pos = 0
    pos = 0.0
    fn_yaml = "parking_spots.yml"

    parking_data = parking_datasets(fn_yaml)
    if parking_data == None:
        import datasets
    parking_data = parking_datasets(fn_yaml)

    cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(cascade_src)
    #car_coor = detect_cars()

    parking_contours, parking_bounding_rects, parking_mask, parking_data_motion = get_parking_info(parking_data)

    kernel_erode = set_erode_kernel()
    kernel_dilate = set_dilate_kernel()
    parking_status, parking_buffer = status(parking_data)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
    
    while True:
        start = time.time()
        #data = requests.get(url, params='frame')
        #print(data.content)
        #data = request.data
        #nparr = np.fromstring(data, np.uint8)
        #simg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #d = json.loads(data.headers)
        first_frame = requestImage()
        frame_pos += 1
        
        #decimg = cv2.imdecode(first_frame,0)
        frame = cv2.resize(first_frame, None, fx=0.6, fy=0.6)
        # Smooth out the image, then convert to grayscale
        blurImg = cv2.GaussianBlur(frame.copy(), (5,5), 3)
        grayImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)
        line_img = frame.copy()
        vpl = np.copy(line_img) * 0 #Virtual Parking Lot

        # Drawing the Overlay. Text overlay at the left corner of screen
        str_on_frame = "%d" % (frame_pos)
        cv2.putText(line_img, str_on_frame, (5,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,255), 2, cv2.LINE_AA)

        


        fgmask = fgbg.apply(blurImg)
        bw = np.uint8(fgmask==255)*255
        bw = cv2.erode(bw, kernel_erode, iterations=1)
        bw = cv2.dilate(bw, kernel_dilate, iterations=1)

        (cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        for c in cnts:
            if cv2.contourArea(c) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(line_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    #Use the classifier to detect cars and help determine which parking spaces are available and unavailable
        detection(parking_bounding_rects, parking_data, parking_status,
                  parking_buffer, grayImg, start, parking_mask, line_img, car_cascade, vpl)
        cv2.imwrite("virtual_parking_lot.jpg",vpl)


        

        # Display video
        cv2.imshow('frame', line_img)
        # cv2.imshow('background mask', bw)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        #elif k == ord('j'):
        #    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos+1000) # jump 1000 frames
        #elif k == ord('u'):
        #    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos + 500)  # jump 500 frames
        #if cv2.waitKey(33) == 27:
        #    break

    cv2.waitKey(0)
    cap.__del__()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
