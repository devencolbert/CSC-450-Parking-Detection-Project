import numpy as np
import cv2
import random
class ImgProcessor(object):

    def __init__(self):
        #setup variable
        #self.
        #network
        self.net = cv2.dnn.readNetFromTensorflow('car_dect/frozen_inference_graph.pb', 'car_dect/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt');
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        #setup parking spots
        self.spots = None

    def process_frame(self, frame):
        cars = self.detect_cars(frame)
        available_spots = self.detect_available(cars, self.spots)

        #return available parking spots(results)

    def detect_cars(self, frame):
        confThreshold = 0.3
        maskThreshold = 0.3
        colorsFile = "car_dect/colors.txt"
        with open(colorsFile, 'rt') as f:
            colorsStr = f.readlines()
        colors = []
        for i in range(len(colorsStr)):
            rgb = colorsStr[i].split(', ')
            color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
            colors.append(color)
        blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
        
 
        # Set the input to the network
        self.net.setInput(blob)
        boxes, masks = self.net.forward(['detection_out_final', 'detection_masks'])
        numClasses = masks.shape[1]
        numDetections = boxes.shape[2]
         
        frameH = frame.shape[0]
        frameW = frame.shape[1]
        
         
        for i in range(numDetections):
            coors = []
            box = boxes[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > confThreshold:
                classId = int(box[1])
                 
                # Extract the bounding box
                left = int(frameW * box[3])
                top = int(frameH * box[4])
                right = int(frameW * box[5])
                bottom = int(frameH * box[6])
                 
                left = max(0, min(left, frameW - 1))
                top = max(0, min(top, frameH - 1))
                right = max(0, min(right, frameW - 1))
                bottom = max(0, min(bottom, frameH - 1))
                # Extract the mask for the object
                classMask = mask[classId]
            
             
            # Draw bounding box, colorize and show the mask on the image
            # Draw a bounding box.
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
             
            # Resize the mask, threshold, color and apply it on the image
            classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
            mask = (classMask > maskThreshold)
            roi = frame[top:bottom+1, left:right+1][mask]
         
            #color = colors[classId%len(colors)]
            #Comment the above line and uncomment the two lines below to generate different instance colors
            colorIndex = random.randint(0, len(colors)-1)
            color = colors[colorIndex]
         
            frame[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
         
            # Draw the contours on the image
            mask = mask.astype(np.uint8)
            retr_tree = cv2.RETR_TREE
            chain_approx_simp = cv2.CHAIN_APPROX_SIMPLE
            contours, hierarchy = cv2.findContours(mask, retr_tree, chain_approx_simp)
            cv2.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)
            coors.append(top)
            coors.append(left)
            coors.append(bottom)
            coors.append(right)
            print(coors)
    
        #detect cars
            return coors

    def detect_available(self, cars, spots):
        pass



        #return

    def format_spots(self, avail):
        pass
        #return all_spots
