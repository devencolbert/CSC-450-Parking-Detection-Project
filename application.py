from flask import Flask, Response, request, redirect, render_template, session, make_response, url_for, abort, send_file
from flask_admin import Admin, BaseView, expose
from flask_sqlalchemy import SQLAlchemy
from flask_admin.contrib.sqla import ModelView
from lib.img_proc import ImgProcessor
from storage.database.models import *
from lib.cam import Camera
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from apscheduler.schedulers.background import BackgroundScheduler

import storage.database.models
import numpy as np
import requests
import atexit
import time
import json
import yaml
import cv2
import os, os.path

location = None
object_id = "id1"
inc = 0

#   Server Configuration

application = Flask(__name__)

application.config.from_object(__name__)
application.config.update(dict(
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(application.root_path + "/storage/database", "database.db"),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    DEBUG = True,
    SECRET_KEY = 'UHu9qUw9SLgKvneuJXmsQRfV',
))

#   Database Configuration

db.init_app(application)
application.app_context().push()
db.create_all()

#   Admin Portal Configuration

class ConfigurationView(BaseView):
    @expose(url='/', methods=('GET', 'POST'))
    def index(self):
        global object_id, location
        form = LoginForm()
        if form.validate_on_submit():
            object_id = form.artist1.data
            location = form.artist2.data
            #print(location.id)
            return redirect('/configuration/display')
        return self.render('configuration.html', form = form)

admin = Admin(application, name = 'Admin Portal', template_mode = 'bootstrap3')

admin.add_view(ConfigurationView(name = 'Configuration', endpoint = 'configuration'))
admin.add_view(ModelView(Lot, db.session, category = "Database"))
admin.add_view(ModelView(Spot, db.session, category = "Database"))
admin.add_view(ModelView(Calculation, db.session, category = "Database"))

#   Flask WTF Forms

class LoginForm(FlaskForm):
    artist1 = StringField('Camera ID', validators=[DataRequired()])
    #artist2 = StringField('Lot ID', validators=[DataRequired()])
    artist2 = QuerySelectField(
        'id',
        query_factory=lambda: Lot.query,
        allow_blank=False
    )
    submit = SubmitField('Submit')

#   Database insert function

def database_import(data, lot):
    with application.app_context():
        #db.session.query(Spot.availability).delete()
        db.session.query(Spot).filter(Spot.lot_location == lot).delete()
        for x in data:
            imported = Spot(availability = x, lot_location = lot)
            db.session.add(imported)
        db.session.commit()
    
#   Scheduler Configuration
    
def update_availability():
    if os.listdir(application.root_path + "/storage/config"):
        path = application.root_path + "/storage/config"
        file_num = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
        #with application.app_context():
            #db.session.query(Spot.availability).delete()
            #db.session.query(Spot.lot_location).delete()
        for x in range(file_num):
            
            #   Retrive frame from video stream via ID
            ident = 'id' + str(x+1)
            #print(ident)
            r = requests.get('http://127.0.0.1:8080/get_frame', headers = {'cam_id': ident})
            data = r.content
            frame = json.loads(data.decode("utf8"))
            frame = np.asarray(frame, np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            
            #   Process frame and return availability information as list
            
            car_dect = ImgProcessor()
            test = ident
            availability_information = car_dect.process_frame(frame, test)
            #print("Success!")
            #print(availability_information[1])
            database_import(availability_information[0], availability_information[1])
    else:
        print("There is no file")

scheduler = BackgroundScheduler()
scheduler.add_job(func = update_availability, trigger = "interval", seconds = 30)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

#   Index Route

@application.route('/')
@application.route('/index')
def index():
    #data = db.session.execute("SELECT * FROM Lot").fetchall()
    data = db.session.execute("SELECT * FROM Lot").fetchall()
    return render_template('index.html', data = data)

#   Info Route

@application.route('/info/<location>')
def info(location):
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
    global inc, location
    stuff = {'id': 0, 'lot': 0, 'points': []}
    arr = []
    data = request.json
    if data != None:
        print(data[0])
        stuff['points'] = [list(data[0]), list(data[1]), list(data[2]), list(data[3])]
        stuff['id'] = inc
        stuff['lot'] = location.id
        inc = inc + 1
        arr.append(stuff)
        with open('./storage/config/' + object_id + '.yml','a') as yamlfile:
            yaml.dump(arr, yamlfile)
        
    return render_template("test.html")

if __name__ == '__main__':
    application.run(host = '127.0.0.1', port = '8090', debug = True)
