def detect_motion(parking_data_motion, grayImg, kernel_erode, kernel_dilate, car_cascade, info, data):
    if parking_data_motion != []:
            for ind, park_coord in enumerate(parking_data_motion):
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

#Still detect if parking spaces are available or not when cars are in motion
detect_motion(parking_data_motion, grayImg, kernel_erode, kernel_dilate, car_cascade, info, data)
