import numpy as np
import cv2
import random as rng
import imutils

img = cv2.imread('test_p.jpg')

#kernel = np.ones((5,5), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
denoised = cv2.fastNlMeansDenoising(imgray, None, 25, 20)
ret,thresh = cv2.threshold(denoised,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, 450, 400, apertureSize = 7)
gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
img2 = np.copy(gradient)
dilation = cv2.dilate(gradient,kernel,iterations = 3)
#erosion = cv2.erode(thresh, kernel, iterations = 1)
dilation2 = cv2.dilate(img2,kernel,iterations = 2)
#result = dilation - dilation2

contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
img3 = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
#img3 = np.copy(img) * 0
line_img = np.copy(img) * 0

cnt = contours[0]

perimeter = cv2.arcLength(cnt,True)
area = cv2.contourArea(cnt)
print(perimeter)
print(area)
print(cnt)

minRect = [None]*len(contours)
minEllipse = [None]*len(contours)
for i, c in enumerate(contours):
    minRect[i] = cv2.minAreaRect(c)
    if c.shape[0] > 5:
        minEllipse[i] = cv2.fitEllipse(c)
    # Draw contours + rotated rects + ellipses

    
for i, c in enumerate(contours):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    # contour
    #cv2.drawContours(img3, contours, i, color)
    # ellipse
    if c.shape[0] > 5:
        #cv2.ellipse(img3, minEllipse[i], color, 2)
        # rotated rectangle
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(img3, [box], 0, color)

for i, c in enumerate(contours):
    if cnt.any() > 3:
        img3 = cv2.drawContours(img3, contours, -1, (0, 100, 0), 1)
    else:
        img3 = cv2.drawContours(img3, contours, -1, (0, 0, 255), 1)

#img3 = cv2.drawContours(img3, contours, -1, (0, 0, 255), 1)

'''
imgray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, 450, 400, apertureSize = 3)

lines = cv2.HoughLinesP(edges,1,np.pi/180,5,10,0)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),2)
'''

cv2.imshow("w", img3)
cv2.waitKey(0)
