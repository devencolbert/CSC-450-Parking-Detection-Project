from flask import Flask, render_template, Response
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
	print(request.header)
	#STEP-1: Locate specific camera
	#STEP-2: Pull-norm-package frame
	#STEP-3: return frame
"""return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
"""

def server_init():
	pass
	#STEP-01: Setup camera dictionary (global obj)
	#STEP-02: Check if cams connected
	#done.
	
if __name__ == '__main__':
	server_init()
	app.run(host='0.0.0.0', port='8000', debug=True)
	
#STEP 1: Load in, convert, and decompress frame for use
frame = ujson.loads(data.decode("utf8"))
frame = numpy.asarray(frame, numpy.uint8)
frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)