from flask import Flask, render_template, request, make_response, Response
from lib.cam import Camera
import json
import numpy
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame = get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame)

@app.route('/get_frame', methods = ['POST', 'GET'])
def get_frame():
	print(request.headers)
	#STEP-1: Locate specific camera
	c = Camera('live',0,cam_dict['cam_1'])
	if request.method == 'GET':
		addr = request.form
	print(addr)
	#STEP-2: Pull-norm-package frame
	frame = c.package(c.norm_frame(c.get_frame()))
	#STEP-3: return frame
	return frame
"""return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
"""

def server_init():
	#pass
	#STEP-01: Setup camera dictionary (global obj)
	global cam_dict
	cam_dict = {'cam_1': 0, 'cam_2': 1, 'cam_3': 2, 'cam_4': 3}
	#STEP-02: Check if cams connected
	c = Camera('live',0,cam_dict['cam_1'])
	print(c.cam_open())
	c = Camera('live',1,cam_dict['cam_2'])
	print(c.cam_open())
	c = Camera('live',2,cam_dict['cam_3'])
	print(c.cam_open())
	c = Camera('live',3,cam_dict['cam_4'])
	print(c.cam_open())
	
if __name__ == '__main__':
	server_init()
	app.run(host='0.0.0.0', port='8000', debug=True)
	
#STEP 1: Load in, convert, and decompress frame for use
frame = json.loads(data.decode("utf8"))
frame = numpy.asarray(frame, numpy.uint8)
frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
