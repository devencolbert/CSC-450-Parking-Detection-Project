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
		available_spots = self.detect_available(spots,frame)
		return available_spots

		#return available parking spots(results)

	def detect_cars(self, frame):
		pass
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
				contours, hierarchy = cv2.findContours(mask, retr_tree, chain_approx_simp)
				cv2.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)
				#while True:
					#cv2.imshow('im', frame)
					#if cv2.waitKey(1) == ord('q'):
						#break
				#cv2.destroyAllWindows()
				coors.append(top)
				coors.append(left)
				coors.append(bottom)
				coors.append(right)
				#print(coors)

		#detect cars
		return coors'''

	def detect_available(self, spots, frame):
		#For right now this gives an error:  carCoor_x1 = car_coor[index]['coors'][key][0]
		#TypeError: list indices must be integers or slices, not dict
		#Spot Coordinates still need to be implemented

		avail = []
		blurImg = cv2.GaussianBlur(frame.copy(), (5,5), 3)
		grayImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)
		
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
				
		spots_change = 0
    # detecting cars and vacant spaces
		for ind, park in enumerate(spots):
			points = np.array(park['points'])
			rect = parking_bounding_rects[ind]
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
			
			total_spots = len(spots)
			for ind, park in enumerate(spots):
				points = np.array(park['points'])
				if parking_status[ind]:
					color = (0,255,0)
					spots_change += 1
					spot = 'Available'
					rect = parking_bounding_rects[ind]
					roi_gray_ov = grayImg[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
					cars = self.cas.detectMultiScale(roi_gray_ov, 1.1, 1)
					if cars == ():
						res = False
					else:
						res = True
					if res:
						parking_data_motion.append(spots[ind])
						
						color = (0,0,255)
				else:
					color = (0,0,255)
					spot = 'Unavailable'
				
				#cv2.drawContours(line_img, [points], contourIdx=-1,color=color, thickness=2, lineType=cv2.LINE_8)
				#cv2.drawContours(vpl, [points], contourIdx=-1,color=color, thickness=2, lineType=cv2.LINE_8)
			avail.append(spots)
		return avail
				
				#moments = cv2.moments(points)
				#centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
		'''for ind, park in enumerate(spots):
			points = np.array(park['points'])
			#print(points)
			xA = max(cars[1], points[0][0])
			yA = max(cars[0], points[0][1])
			xB = min(cars[3], points[3][0])
			yB = min(cars[2], points[3][1])

			interArea = (xB - xA + 1) * (yB - yA + 1)
			carBoxArea = (cars[3] - cars[1] + 1) * (cars[2] - cars[0] + 1)
			spotBoxArea = (points[3][0] - points[0][0] + 1) * (points[3][1] - points[0][1] + 1)

			iou = abs(interArea / float(carBoxArea + spotBoxArea - interArea))
			#print(iou)
			if iou < 1 and iou > 0:
				avail.append('unavailable')
			else:
				avail.append('available')
		return avail

		#for index in range(len(car_coor)):
		#    for key in car_coor[index]:
		#        print(car_coor[index][key])
		#Assigning each car coordinate to a variable
		for index, key in enumerate(car_coor):
			carCoor = car_coor[index]['coors']
			carCoor_x1 = car_coor[index]['coors'][key][0]
			carCoor_y1 = car_coor[index]['coors'][key][1]
			carCoor_x2 = car_coor[index]['coors'][key+1][0]
			carCoor_y2 = car_coor[index]['coors'][key+1][1]
		#print(carCoor_x1)


		#for i in range(len(spot_Coor)):
		#    for k in spot_Coor[i]:
				#print(spot_Coor[i][k])
		#Assigning each parking spot coordinate to variable
		spotCoor = [spot_Coor[0]['points'][2], spot_Coor[0]['points'][0]]
		spotCoor_x1 = spot_Coor[0]['points'][2][0]
		spotCoor_y1 = spot_Coor[0]['points'][2][1]
		spotCoor_x2 = spot_Coor[0]['points'][0][0]
		spotCoor_y2 = spot_Coor[0]['points'][0][1]
		#print(spotCoor_x2)

		
		xA = max(carCoor_x1, spotCoor_x1)
		yA = max(carCoor_y1, spotCoor_y1)
		xB = min(carCoor_x2, spotCoor_x2)
		yB = min(carCoor_y2, spotCoor_y2)

		interArea = (xB - xA + 1) * (yB - yA + 1)
		carBoxArea = (carCoor_x2 - carCoor_x1 + 1) * (carCoor_y2 - carCoor_y1 + 1)
		spotBoxArea = (spotCoor_x2 - spotCoor_x1 + 1) * (spotCoor_y2 - spotCoor_y1 + 1)

		iou = interArea / float(carBoxArea + spotBoxArea - interArea)

		return iou'''




		#return

	def format_spots(self, avail):
		pass
		#return all_spots
