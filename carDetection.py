import numpy as np
import cv2 as cv


car_cascade = cv.CascadeClassifier('cascade.xml')


img = cv.imread('toyCars2.jpg')
newImg = cv.resize(img, (480,640))
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.cvtColor(newImg, cv.COLOR_BGR2GRAY)

cars = car_cascade.detectMultiScale(gray, 1.01, 1)
for(x,y,w,h) in cars:
    #cv.rectangle(img,(x,y), (x+w, y+h), (255, 0, 0), 1)
	cv.rectangle(newImg,(x,y), (x+w, y+h), (255, 0, 0), 1)

#cv.imshow('img', img)
cv.imshow('img', newImg)
cv.waitKey(0)
cv.destroyAllWindows()