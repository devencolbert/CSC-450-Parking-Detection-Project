from flask import Flask, render_template, request, make_response, Response
from lib.cam import Camera
import json
import numpy
import cv2
FEEDS = {} #Format: {"<ID>: Camera Object"}
SETUP = False
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_frame', methods = ['GET'])
def get_frame():
	print(request.headers)
	
	#STEP-1: Locate specific camera
	
	#STEP-2: Pull-norm-package frame
	#frame = c.package(c.norm_frame(c.get_frame()))
	
	#STEP-3: return frame
	return frame
	

def init():
	global FEEDS
	#STEP-01: Build the Feeds Dict from user input
	
	#STEP-02: Check video feeds
	
	#STEP-03: return True/False if setup failed
	input('Press Enter to continue...')

if __name__ == '__main__':
	init()
	app.run(host='127.0.0.1', port='8080', debug=False)
