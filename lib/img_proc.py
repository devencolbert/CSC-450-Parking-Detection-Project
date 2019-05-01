#from car_dect.new_segment import *
import numpy as np
import cv2
import random
import yaml
import time

class ImgProcessor(object):

        def __init__(self):
                #setup variable
                #self.
                #network
                #self.net = cv2.dnn.readNetFromTensorflow('lib/car_dect/frozen_inference_graph.pb', 'lib/car_dect/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt');
                #self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.cars = 'cars.xml'
                self.cas = cv2.CascadeClassifier(self.cars)
                #setup parking spots
                with open('id2.yml', 'r') as stream:
                        self.spots = yaml.load(stream)
                        

        def process_frame(self, frame):
                #cars = self.detect_cars(frame)
                spots = self.spots
                cas = self.cas
                parking_contours, parking_bounding_rects, parking_mask, parking_data_motion = self.get_parking_info(spots)
                kernel_erode = self.set_erode_kernel()
                kernel_dilate = self.set_dilate_kernel()
                parking_status, parking_buffer = self.status(spots)
                start = time.time()
                frame = cv2.GaussianBlur(frame.copy(), (5,5), 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                available_spots = self.detection(parking_bounding_rects, spots, parking_status,
                  parking_buffer, frame, start, parking_mask, cas)
                return available_spots

                #return available parking spots(results)
                
        def run_classifier(self, img, id, car_cascade):
                cars = car_cascade.detectMultiScale(img, 1.1, 1)
                if cars == ():
                        return False
                else:
                        return True
                        
        def get_parking_info(self, spots):    
                parking_contours = []
                parking_bounding_rects = []
                parking_mask = []
                parking_data_motion = []
                if spots != None:
                        for park in spots:
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

        def set_erode_kernel(self):
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                return kernel_erode

        def set_dilate_kernel(self):
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(5,19))
                return kernel_dilate

        def status(self, spots):
                if spots != None:
                        parking_status = [False]*len(spots)
                        parking_buffer = [None]*len(spots)
                return parking_status, parking_buffer;
                
        def print_parkIDs(self, park, points, cas, parking_bounding_rects, frame, spots, parking_status):

                avail = []
                spots_change = 0
                total_spots = len(spots)
                for ind, park in enumerate(spots):
                        points = np.array(park['points'])
                        if parking_status[ind]:
                                color = (0,255,0)
                                spots_change += 1
                                spot = 'Available'
                                avail.append(spot)
                                rect = parking_bounding_rects[ind]
                                roi_gray_ov = frame[rect[1]:(rect[1] + rect[3]),
                                                           rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
                                #res = run_classifier(roi_gray_ov, ind, car_cascade)
                                #if res:
                                #    parking_data_motion.append(spots[ind])
                                        
                                #    color = (0,0,255)
                        else:
                                color = (0,0,255)
                                spot = 'Unavailable'
                                avail.append(spot)
                return avail
                                
        def detection(self, parking_bounding_rects, spots, parking_status,
                  parking_buffer, frame, start, parking_mask, cas):

                
                spots_change = 0
                '''coors = []
                car_rect = []
                car_bounding_rects = get_car_info(car_coor)
                for ind, coor in enumerate(car_coor):
                        coors = np.array(coor['coors'])
                        car_rect = car_bounding_rects[ind]'''
                # detecting cars and vacant spaces
                for ind, park in enumerate(spots):
                        points = np.array(park['points'])
                        rect = parking_bounding_rects[ind]

                        '''if points[2].all() > coors[1].all() or points[0].all() > coors[0].all():
                                #print(points)
                                overlap = False
                        else:
                                overlap = True'''
                        roi_gray = frame[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])] # crop roi for faster calcluation

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
                
                        avail = self.print_parkIDs(park, points, cas, parking_bounding_rects, frame, spots, parking_status)
                        return avail

        def detect_cars(self, frame):
                pass

        def detect_available(self, spots, frame):
                pass



                #return

        def format_spots(self, avail):
                pass
                #return all_spots

