#from lib.car_dect.new_segment import *
import numpy as np
import cv2
import random
import yaml
import os

class ImgProcessor(object):

	def __init__(self):
		#setup variable
		#self.
		#network
		#self.net = cv2.dnn.readNetFromTensorflow('lib/car_dect/frozen_inference_graph.pb', 'lib/car_dect/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt');
		#self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		#self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
		#setup parking spots
		#with open('id2.yml', 'r') as stream:
				#self.spots = yaml.load(stream)
		with open('./lib/car_dect/' + 'coors.yml', 'r') as stream:
			self.cars = yaml.load(stream)
		#self.cars = coor


	def process_frame(self, frame, config):
		coors = self.cars
		cars = self.detect_cars(frame, coors)
		#spots = self.spots
		spots = self.get_config(config)
		available_spots = self.detect_available(cars, spots)
		return available_spots

		#return available parking spots(results)

	def get_config(self, config):
		with open('./storage/config/' + config + '.yml', 'r') as stream:
				self.spots = yaml.load(stream)
		return self.spots

	def detect_cars(self, frame, coors):
		for ind, car in enumerate(coors):
			points = np.array(car['coors'])
			#print(points)
			return points
		#pass
		'''confThreshold = 0.5
		maskThreshold = 0.3

		classesFile = "lib/car_dect/mscoco_labels.names";
		classes = None
		with open(classesFile, 'rt') as f:
		   classes = f.readlines()
		
		colorsFile = "lib/car_dect/colors.txt"
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
		#print(numDetections)
		 
		frameH = frame.shape[0]
		frameW = frame.shape[1]
		left = 0
		top = 0
		right = 0
		bottom = 0
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
				
				if left == 0:
						coors.append(top)
						coors.append(left)
						coors.append(bottom)
						coors.append(right)
						#print(coors)
				else:
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
						im2, contours, hierarchy = cv2.findContours(mask, retr_tree, chain_approx_simp)
						cv2.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)
						while True:
								cv2.imshow('im', frame)
								if cv2.waitKey(1) == ord('q'):
										break
						cv2.destroyAllWindows()
						coors.append(top)
						coors.append(left)
						coors.append(bottom)
						coors.append(right)
						#print(coors)

		#detect cars'''
		

	def detect_available(self, cars, spots):
		#For right now this gives an error:  carCoor_x1 = car_coor[index]['coors'][key][0]
		#TypeError: list indices must be integers or slices, not dict
		#Spot Coordinates still need to be implemented

		avail = []
		#lot_id = park['lot']
		#print(lot_id)
		#print("above")
		for ind, park in enumerate(spots):
			#print(ind)
			#print(park)
			points = np.array(park['points'])
			#print(points)
			xA = max(cars[0][1], points[0][0])
			yA = max(cars[0][0], points[0][1])
			xB = min(cars[1][1], points[3][0])
			yB = min(cars[1][0], points[3][1])

			interArea = (xB - xA + 1) * (yB - yA + 1)
			carBoxArea = (cars[1][1] - cars[0][1] + 1) * (cars[1][0] - cars[0][0] + 1)
			spotBoxArea = (points[3][0] - points[0][0] + 1) * (points[3][1] - points[0][1] + 1)

			iou = abs(interArea / float(carBoxArea + spotBoxArea - interArea))
			#print(iou)
			lot_id = park['lot']
			#print(lot_id)
			#print("Should be above")
			if iou < 1 and iou > 0:
				avail.append('Unavailable')
			else:
				avail.append('Available')
		return avail, lot_id

	def format_spots(self, avail):
		pass
		#return all_spots
