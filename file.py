import numpy as np
import cv2

img = cv2.imread('test.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(imgray, (5,5), 0)
img = cv2.Canny(img, 50, 150)
ret,thresh = cv2.threshold(img,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

img = cv2.drawContours(img, contours, -1, (150,255,0), 3)

cv2.imshow("window",img)
cv2.waitKey(0)
