from flask import Flask, render_template, Response
from camera_client import Camera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera_client):
    while True:
        frame = camera_client.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame)

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
'''
@app.route('/video_feed/<camera_type>/<device>')
def video_feed(camera_type, device):
    """Video streaming route. Put this in the src attribute of an img tag."""
    camera_stream = import_module('camera_' + camera_type).Camera
    if camera_type == 'opencv':
        camera_stream.set_video_source(int(device))
    return Response(gen(camera_stream(camera_type=camera_type, device=int(device))),
                    mimetype='multipart/x-mixed-replace; boundary=frame')'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)
