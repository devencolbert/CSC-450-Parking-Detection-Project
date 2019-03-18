import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import os, glob
#import camera_client
#import datasets


def run_classifier(img, id, car_cascade):
    cars = car_cascade.detectMultiScale(img, 1.1, 1)
    if cars == ():
        return False
    else:
        return True


def parking_datasets(fn_yaml):
    with open(fn_yaml, 'r') as stream:
        parking_data = yaml.load(stream)
    return parking_data

def get_parking_info(parking_data):    
    parking_contours = []
    parking_bounding_rects = []
    parking_mask = []
    parking_data_motion = []
    if parking_data != None:
        for park in parking_data:
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

def set_erode_kernel():
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    return kernel_erode

def set_dilate_kernel():
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(5,19))
    return kernel_dilate

def status(parking_data):
    if parking_data != None:
        parking_status = [False]*len(parking_data)
        parking_buffer = [None]*len(parking_data)
    return parking_status, parking_buffer;

def print_parkIDs(park, points, line_img, car_cascade,
                  parking_bounding_rects, grayImg, parking_data, parking_status):
    for ind, park in enumerate(parking_data):
        points = np.array(park['points'])
        if parking_status[ind]:
            color = (0,255,0)
            #spot = 'Available'
            rect = parking_bounding_rects[ind]
            roi_gray_ov = grayImg[rect[1]:(rect[1] + rect[3]),
                           rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
            res = run_classifier(roi_gray_ov, ind, car_cascade)
            if res:
                parking_data_motion.append(parking_data[ind])
                
                color = (0,0,255)
        else:
            color = (0,0,255)
            #spot = 'Unavailable'
        
        cv2.drawContours(line_img, [points], contourIdx=-1,
                             color=color, thickness=2, lineType=cv2.LINE_8)
        moments = cv2.moments(points)
        centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
        # putting numbers on marked regions
        cv2.putText(line_img, str(park['id']), (centroid[0]+1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(line_img, str(park['id']), (centroid[0]-1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(line_img, str(park['id']), (centroid[0]+1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(line_img, str(park['id']), (centroid[0]-1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(line_img, str(park['id']), centroid, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def detection(parking_bounding_rects, parking_data, parking_status,
              parking_buffer, grayImg, pos, parking_mask, line_img, car_cascade):
    #info = {'id': 0, 'availability': ''}
    #file_path = 'parking_spots_info.yml'
    # detecting cars and vacant spaces
    for ind, park in enumerate(parking_data):
        points = np.array(park['points'])
        rect = parking_bounding_rects[ind]
        roi_gray = grayImg[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])] # crop roi for faster calcluation

        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
        
        points[:,0] = points[:,0] - rect[0] # shift contour to roi
        points[:,1] = points[:,1] - rect[1]
        delta = np.mean(np.abs(laplacian * parking_mask[ind]))
        
        
        status = delta < 2.2
        # If detected a change in parking status, save the current time
        if status != parking_status[ind] and parking_buffer[ind]==None:
            parking_buffer[ind] = pos
            change_pos = pos
            
        # If status is still different than the one saved and counter is open
        elif status != parking_status[ind] and parking_buffer[ind]!=None:
            if pos - parking_buffer[ind] > 1:
                parking_status[ind] = status
                parking_buffer[ind] = None
        # If status is still same and counter is open
        elif status == parking_status[ind] and parking_buffer[ind]!=None:
            parking_buffer[ind] = None

# changing the color on the basis on status change occured in the above section and putting numbers on areas
    
        print_parkIDs(park, points, line_img, car_cascade,
                      parking_bounding_rects, grayImg, parking_data, parking_status)


def detect_motion(parking_data_motion, grayImg, kernel_erode, kernel_dilate, car_cascade):
    if parking_data_motion != []:
            for index, park_coord in enumerate(parking_data_motion):
                points = np.array(park_coord['points'])
                color = (0, 0, 255)
                recta = parking_bounding_rects[ind]
                roi_gray1 = grayImg[recta[1]:(recta[1] + recta[3]),
                                recta[0]:(recta[0] + recta[2])]  # crop roi for faster calcluation

                
                fgbg1 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
                roi_gray1_blur = cv2.GaussianBlur(roi_gray1.copy(), (5, 5), 3)

                
                fgmask1 = fgbg1.apply(roi_gray1_blur)
                bw1 = np.uint8(fgmask1 == 255) * 255
                bw1 = cv2.erode(bw1, kernel_erode, iterations=1)
                bw1 = cv2.dilate(bw1, kernel_dilate, iterations=1)

                
                (_, cnts1, _) = cv2.findContours(bw1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # loop over the contours
                for c in cnts1:
                    print(cv2.contourArea(c))
                    # if the contour is too small, we ignore it
                    if cv2.contourArea(c) < 4:
                        continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    classifier_result1 = run_classifier(roi_gray1, index, car_cascade)
                    if classifier_result1:
                    # print(classifier_result)
                        color = (0, 0, 255)  # Red again if car found by classifier
                    else:
                        color = (0,255, 0)
                classifier_result1 = run_classifier(roi_gray1, index)
                if classifier_result1:
                    color = (0, 0, 255)  # Red again if car found by classifier
                else:
                    color = (0, 255, 0)
                cv2.drawContours(line_img, [points], contourIdx=-1,
                                     color=color, thickness=2, lineType=cv2.LINE_8)

            # detect people in the image. Slows down the program, requires high GPU speed
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            # draw the  bounding boxes
            for (x, y, w, h) in rects:
                cv2.rectangle(line_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
   
    
def main():

    fn = "output.mp4"
    fn_yaml = "parking_spots.yml"

    parking_data = parking_datasets(fn_yaml)
    if parking_data == None:
        import datasets

    cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(cascade_src)
    global_str = "Last change at: "
    change_pos = 0.00
    data = []

    #video info for processing the footage
    cap = cv2.VideoCapture(fn)
    video_info = {  'fps':    cap.get(cv2.CAP_PROP_FPS),
                    'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.6),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.6),
                    'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                    'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    parking_contours, parking_bounding_rects, parking_mask, parking_data_motion = get_parking_info(parking_data)

    kernel_erode = set_erode_kernel()
    kernel_dilate = set_dilate_kernel()
    parking_status, parking_buffer = status(parking_data)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
    
    while(cap.isOpened()):
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Current position of the video file in seconds
        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES) # Index of the frame to be decoded/captured next
        ret, first_frame = cap.read()
        if ret == True:
            frame = cv2.resize(first_frame, None, fx=0.6, fy=0.6)
        if ret == False:
            print("Video ended")
            break

    # Smooth out the image, then convert to grayscale
        blurImg = cv2.GaussianBlur(frame.copy(), (5,5), 3)
        grayImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)
        line_img = frame.copy()

        # Drawing the Overlay. Text overlay at the left corner of screen
        str_on_frame = "%d/%d" % (frame_pos, video_info['num_of_frames'])
        cv2.putText(line_img, str_on_frame, (5,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,255), 2, cv2.LINE_AA)


        fgmask = fgbg.apply(blurImg)
        bw = np.uint8(fgmask==255)*255
        bw = cv2.erode(bw, kernel_erode, iterations=1)
        bw = cv2.dilate(bw, kernel_dilate, iterations=1)

        (cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        for c in cnts:
            if cv2.contourArea(c) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(line_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    #Use the classifier to detect cars and help determine which parking spaces are available and unavailable
        detection(parking_bounding_rects, parking_data, parking_status,
                  parking_buffer, grayImg, pos, parking_mask, line_img, car_cascade)
    #Still detect if parking spaces are available or not when cars are in motion
        detect_motion(parking_data_motion, grayImg, kernel_erode, kernel_dilate, car_cascade)

        

        # Display video
        cv2.imshow('frame', line_img)
        # cv2.imshow('background mask', bw)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('j'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos+1000) # jump 1000 frames
        elif k == ord('u'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos + 500)  # jump 500 frames
        if cv2.waitKey(33) == 27:
            break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
