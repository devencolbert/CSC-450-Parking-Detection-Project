import numpy as np
import cv2

img = cv2.imread('test3.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(imgray,9,75,75)
ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, 300, 350, apertureSize = 3)

img2 = cv2.imread('test.jpg')
imgray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
blur2 = cv2.bilateralFilter(imgray2,9,75,75)
edges2 = cv2.Canny(blur2, 300, 350, apertureSize = 3)

line_img = np.copy(img) * 0





minLineLength = 100
maxLineGap = 0
lines = cv2.HoughLinesP(edges,1,np.pi/180,0,minLineLength,maxLineGap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),1)




ret2,thresh2 = cv2.threshold(edges2,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
'''
M = cv2.moments(cnt)
print(M)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print(cx)
print(cy)

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(img,[box],0,(0,255,0),2)
'''
#img = cv2.drawContours(img, contours, -1, (150,255,0), 1)
#img2 = cv2.drawContours(img2, contours2, -1, (150,255,0), 1)
img3 = cv2.addWeighted(img, 0.8, line_img, 1, 0)

cv2.imshow("window",line_img)
#cv2.imshow("window2",thresh)
cv2.waitKey(0)
