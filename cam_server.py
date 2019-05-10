from flask import Flask, render_template, request, Response
from lib.cam import Camera
import json
import numpy as np
import cv2

#    Global Variables
#    FEEDS format: {"<ID>: Camera Object"}

FEEDS = {}
app = Flask(__name__)

#    Index route [FOR TESTING]

@app.route('/')
def cam_index():
	return render_template('cam_index.html')

#    Retrieve frame based on header input

@app.route('/get_frame', methods = ['GET'])
def get_frame():

	# Retrive camera ID from HTTP headers
	# Extract, normalize, and package frame
	cam_id = request.headers['cam_id']
	frame = FEEDS[cam_id].get_frame()
	
	# If camera feed is unavailable, replace frame with "Unavailable" frame
	if np.shape(frame) == ():
		frame = np.zeros([480, 640], dtype = np.uint8)
		frame.fill(100)
		label = str(cam_id) + ' is OFFLINE'
		cv2.putText(frame, label, (240, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
		frame = FEEDS[cam_id].package((FEEDS[cam_id].norm_frame(frame)))
	else:
		frame = FEEDS[cam_id].package((FEEDS[cam_id].norm_frame(frame)))
		
	return frame

#    User inputs for feed setup
#    Loop until user quits initilization

def camera_import(isTrue):
        id_increment = 0
        while isTrue:
                # Video feed object setup [Mode, Addr, ID]
                i = 'ID'
                mode = input('\nIs the video feed from camera "LIVE" or from a "FILE": ')
                mode = mode.lower()
                if (mode == 'live'):
                        addr = input('Enter the address of the live video as an integer: ')
                        addr = int(addr)
                elif (mode == 'file'):
                        addr = input('Enter the file name of the video: ')
                else:
                        print('\nYour input was incorrect. Enter either "LIVE" or "FILE".\n')
                        camera_import(isTrue)
                cam_id = input('Enter an ID for video feed: ')
                FEEDS[i+str(id_increment)] = Camera(mode, addr, cam_id)

                # Print current list of videos
                # Prompt user to quit or continue
                print('\nCurrent feeds:')
                print(FEEDS)
                ans = input('\nEnter "Q" to quit or "ENTER" to continue initialization: ')
                ans = ans.lower()
                if ans == 'q':
                        isTrue = False
                id_increment += 1
        return FEEDS

#    Initialize video feed server

def init():
	global FEEDS
	
	# Build the Feeds dictionary from user input
	x = True
	FEEDS = camera_import(x)
	
	# Verify that video feeds are initialized
	if len(FEEDS) == 2:
		video_feeds = np.hstack((FEEDS['ID0'].get_frame(),FEEDS['ID1'].get_frame()))
	else:
		video_feeds = FEEDS['ID0'].get_frame()

	# Display video feed frames until user exits
	# Prompt user to finish init process
	while(True):
		cv2.imshow("frame", video_feeds)
		if cv2.waitKey(1) == ord('q'):
			break
	cv2.destroyAllWindows()
	input('\nPress Enter to finish initialization...')

if __name__ == '__main__':
	init()
	app.run(host='0.0.0.0', port='8080', debug = False)
