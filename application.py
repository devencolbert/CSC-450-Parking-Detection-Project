from flask import Flask, Response, request, render_template, session, make_response, url_for, abort, send_file
from flask_admin import Admin, BaseView, expose
from flask_sqlalchemy import SQLAlchemy
from flask_admin.contrib.sqla import ModelView
from storage.database.models import *
from lib.cam import Camera
import numpy as np
import requests
import json
import cv2
import os

FEEDS = {}
application = Flask(__name__)

application.config.from_object(__name__)
application.config.update(dict(
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(application.root_path + "/storage/database", "app.db"),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    DEBUG = False,
    SECRET_KEY = 'development-key',
))

db.init_app(application)
application.app_context().push()
db.create_all()

admin = Admin(application, name = 'Admin Portal', template_mode = 'bootstrap3')

class ConfigurationView(BaseView):
    @expose('/')
    def index(self):
        return self.render('configuration.html')

admin.add_view(ConfigurationView(name='Configuration', endpoint='configuration'))
admin.add_view(ModelView(Lot, db.session, category="Database"))
admin.add_view(ModelView(Spot, db.session, category="Database"))

@application.route('/')
@application.route('/index')
def index():
    lot = []
    #Create connection session between SQLAlchemy database and server
    #Select ALL records in tables Lot
    #Store queries in data and push to INDEX template
    data = db.session.execute("SELECT * FROM Lot").fetchall()
    #Lot.query.all()
    #for l in Lot.query.all():
        #lot.append(l.location)
    return render_template('index.html', data = data)
    #return json.dumps(lot)

@application.route('/info/<location>')
def info(lot_id):
    lot_location = location
    #Create connection session between SQLAlchemy database and server
    #Select records in table Spot based on LOT_ID parameter
    #Store queries in data and push to INFO template
    #data = db.session.execute("SELECT * FROM Spot WHERE location = :lot_location;", {"lot_location": lot_location}).fetchall()
    Spot.query.filter()
    return render_template('info.html', data = data)

@application.route('/test_input', methods = ['GET', 'POST'])
def test():
    global FEEDS
    x = True
    increment = 0
    while x:
        i = 'ID'
        mode = input('Is the video feed from camera "live" or from a "file": ')
        if mode == 'live':
            addr = input('Enter the address of the live video as a integer: ')
            addr = int(addr)
        else:
            addr = input('Enter the file name of the video: ')
        cam_id = input('Enter in camera ID for video feed: ')
        print(mode)
        print(addr)
        FEEDS[i+str(increment)] = Camera(mode, addr, cam_id)
        print(FEEDS)
        ans = input('Enter "q" to quit: ')
        if ans == 'q':
            x = False
        increment += 1
    return "This page will show video feeds eventually"

@application.route('/test_config', methods=['GET'])
def test_config():
    #var = input('Enter ID: ')
    #print(FEEDS[var])
    #frame = FEEDS[var].package(FEEDS[var].norm_frame(FEEDS[var].get_frame()))
    #print(frame)
	r = requests.get('http://127.0.0.1:8080/get_frame').content
	print(r)
	frame = json.loads(r)
	frame = np.asarray(frame, np.uint8)
	frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
	r, jpg = cv2.imencode('.jpg', frame)
	#response = make_response(jpg.tobytes())
    #return Response(FEEDS[var])
	return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
	#return response

# Accept input for Camera Object ID and Address
# Timed JSON call (Loop Image Processing Software)

# Call Process_Frame
# Begin image processing as seperate thread
# When requested to pull frames, get ID and that calls requests

if __name__ == '__main__':
    application.run(host = '127.0.0.1', port = '8090', debug = False)
