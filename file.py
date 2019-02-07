import numpy as np
import cv2

img = cv2.imread('test3.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(imgray,9,75,75)
edges = cv2.Canny(blur, 50, 150, apertureSize = 3)
'''
minLineLength = 100
maxLineGap = 100
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

'''
ret,thresh = cv2.threshold(edges,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[4]


rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img,[box],0,(0,255,0),2)

#img = cv2.drawContours(img, [cnt], 0, (150,255,0), 3)

cv2.imshow("window",img)
cv2.waitKey(0)
