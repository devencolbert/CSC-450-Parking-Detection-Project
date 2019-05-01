from flask import Flask, Response, request, redirect, render_template, session, make_response, url_for, abort, send_file
from flask_admin import Admin, BaseView, expose
from flask_sqlalchemy import SQLAlchemy
from flask_admin.contrib.sqla import ModelView
from storage.database.models import *
from lib.cam import Camera

from flask_wtf import FlaskForm
from wtforms import StringField, SelectField
from wtforms.validators import DataRequired
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from apscheduler.schedulers.background import BackgroundScheduler

import storage.database.models
import numpy as np
import requests
import json
import yaml
import cv2
import os
import time
import atexit
from lib.img_proc import ImgProcessor

location = None
object_id = "id1"
inc = 0
application = Flask(__name__)

application.config.from_object(__name__)
application.config.update(dict(
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(application.root_path + "/storage/database", "app.db"),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    DEBUG = False,
    SECRET_KEY = 'UHu9qUw9SLgKvneuJXmsQRfV',
))

#   Database Config

db.init_app(application)
application.app_context().push()
db.create_all()

#   Admin Portal

class ConfigurationView(BaseView):
    @expose(url='/', methods=('GET', 'POST'))
    def index(self):
        global object_id, location
        form = LoginForm()
        if form.validate_on_submit():
            object_id = form.artist1.data
            location = form.artist2.data
            return redirect('/configuration/display')
        return self.render('configuration.html', form = form)

admin = Admin(application, name = 'Admin Portal', template_mode = 'bootstrap3')

admin.add_view(ConfigurationView(name='Configuration', endpoint='configuration'))
admin.add_view(ModelView(Lot, db.session, category="Database"))
admin.add_view(ModelView(Spot, db.session, category="Database"))

#   Admin Portal Forms

class LoginForm(FlaskForm):
    artist1 = StringField('Camera ID', validators=[DataRequired()])
    #artist2 = SelectField('Location', coerce=int)
    artist2 = QuerySelectField(
        'Location',
        query_factory=lambda: Lot.query,
        allow_blank=False
    )
    
#   Scheduler Test
    
def car_detect():
	r = requests.get('http://127.0.0.1:8080/get_frame', headers= {'cam_id': object_id})
	data = r.content
	frame = json.loads(data.decode("utf8"))
	frame = np.asarray(frame, np.uint8)
	frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

	car_dect = ImgProcessor()
	car = car_dect.process_frame(frame)
	print(car)
	
	with application.app_context():
		db.session.query(Spot.availability).delete()
		i = 1
		for x in car:
			u = Spot(spot_id = i, availability = x, lot_location = object_id)
			i += 1
			db.session.add(u)
			db.session.commit()


scheduler = BackgroundScheduler()
scheduler.add_job(func = car_detect, trigger = "interval", seconds = 10)
scheduler.start()

atexit.register(lambda: scheduler.shutdown()) # Shutdown scheduler

#   Index Route

@application.route('/')
@application.route('/index')
def index():
    data = db.session.execute("SELECT * FROM Lot").fetchall()
    return render_template('index.html', data = data)

#   Info Route

@application.route('/info/<location>')
def info(lot_id):
    data = db.session.execute("SELECT * FROM Spot WHERE lot_location = :lot_location;", {"lot_location": location}).fetchall()
    return render_template('info.html', data = data)

#   Test Config Route

@application.route('/get_frame', methods=['GET'])
def get_frame():
    r = requests.get('http://127.0.0.1:8080/get_frame', headers = {"cam_id": object_id}).content
    frame = json.loads(r)
    frame = np.asarray(frame, np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    r, jpg = cv2.imencode('.jpg', frame)
    return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

#@application.route('/test', methods = ['GET', 'POST'])
#def test():
    #global object_id
    #form = LoginForm()
    #if form.validate_on_submit():
        #object_id = form.artist1.data
        #print(object_id)
        #return redirect('/display')
    #return render_template('configuration.html', form = form)

@application.route('/configuration/display', methods = ['GET', 'POST'])
def display():
    global inc
    stuff = {'id': 0, 'points': []}
    arr = []
    data = request.json
    if data != None:
        print(data[0])
        stuff['points'] = [list(data[0]), list(data[1]), list(data[2]), list(data[3])]
        stuff['id'] = inc
        inc = inc + 1
        arr.append(stuff)
        with open(object_id + '.yml','a') as yamlfile:
            yaml.dump(arr, yamlfile)
        
    return render_template("test.html")

if __name__ == '__main__':
    application.run(host = '127.0.0.1', port = '8090', debug = False)

