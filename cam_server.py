from flask import Flask, render_template, request, Response
from lib.cam import Camera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera_client):
    while True:
        frame = camera_client.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame)

@app.route('/get_frame')
def get_frame():
	print(request.headers)
	#STEP-1: Locate specific camera
	c = Camera('live',0,cam_dict['cam_1'])
	#STEP-2: Pull-norm-package frame
	frame = c.package(c.norm_frame(c.get_frame()))
	#STEP-3: return frame
	return frame
"""return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
"""

def server_init():
	pass
	#STEP-01: Setup camera dictionary (global obj)
	global cam_dict
	cam_dict = {'cam_1': 1, 'cam_2': 2, 'cam_3': 3, 'cam_4': 4}
	#STEP-02: Check if cams connected
	
	#done.
	
if __name__ == '__main__':
	server_init()
	app.run(host='0.0.0.0', port='8000', debug=True)
	
#STEP 1: Load in, convert, and decompress frame for use
frame = ujson.loads(data.decode("utf8"))
frame = numpy.asarray(frame, numpy.uint8)
frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)