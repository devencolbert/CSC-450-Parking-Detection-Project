import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import os, glob
import camera_client

#moduleName = input('Enter module name: ')
#importlib.import_module(moduleName)

def all_images(images, cmap=None):
    cols = 3
    rows = (len(images)+1)//cols

    plt.figure(figsize=(15, 12))

    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        #cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

images = [plt.imread(path) for path in glob.glob('test/*.jpg')]
#all_images(images)

def mask_images(image):
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,200,0])
    upper_white = np.array([255,255,255])
    white = cv2.inRange(image, lower_white, upper_white)
    
    lower_yellow = np.array([10,0,100])
    upper_yellow = np.array([255,255,255])
    yellow = cv2.inRange(image, lower_yellow, upper_yellow)
    
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.bitwise_and(image, image, mask=mask)
    return mask

mask_image = list(map(mask_images, images))
#all_images(mask_image)

def grayScale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray_images = list(map(grayScale, mask_image))
#all_images(gray_images)

def blur_image(image):
    return cv2.GaussianBlur(image, (5,5), 0)

blur = list(map(blur_image, gray_images))
#all_images(blur)

def canny_detection(image, low =50, high=150):
    return cv2.Canny(image, low, high)

edge = list(map(lambda image: canny_detection(image), blur))
#all_images(edge)

def lines(image):
    return cv2.HoughLinesP(image, 1, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=100)

parking_lines = list(map(lines, edge))

def draw(image,linez):
    draw_lines = np.copy(image)

    for line in linez:
        for x1, y1, x2, y2 in line:
            cv2.line(draw_lines, (x1,y1),(x2,y2),(255,0,0),5)
    return draw_lines

line_images = []
for image, linez in zip(images, parking_lines):
    line_images.append(draw(image,linez))
    
all_images(line_images)
'''
def contour(image):
    contour_lines = np.copy(image)

    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    cv2.drawContours(contour_lines, approx, -1 (0,255,0),3)

rect = list(map(contour, line_images))
'''
