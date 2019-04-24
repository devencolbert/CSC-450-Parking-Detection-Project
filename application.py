from flask import Flask, Response, request, render_template, session, make_response, url_for, abort, send_file
from flask_admin import Admin, BaseView, expose
from flask_sqlalchemy import SQLAlchemy
from flask_admin.contrib.sqla import ModelView
from storage.database.models import *
from lib.cam import Camera
import numpy as np
import requests
import json
import yaml
import cv2
import os

inc = 0
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
        #if request.method == 'POST':
        data = request.json
        print(data)
    #with open('config.yml','r') as yamlfile:
        #cur_yaml = yaml.load(yamlfile)
        #cur_yaml.extend(data)
        #print(cur_yaml)

    #with open('config.yml','w') as yamlfile:
        #yaml.dump(data, yamlfile, default_flow_style=False)
                       
        #return render_template("test.html")
        return self.render('configuration.html')

admin.add_view(ConfigurationView(name='Configuration', endpoint='configuration'))
admin.add_view(ModelView(Lot, db.session, category="Database"))
admin.add_view(ModelView(Spot, db.session, category="Database"))

@application.route('/')
@application.route('/index')
def index():
    #   Create connection session between SQLAlchemy database and server
    #   Select ALL records in tables Lot
    #   Store queries in data and push to INDEX template
    data = db.session.execute("SELECT * FROM Lot").fetchall()
    return render_template('index.html', data = data)

@application.route('/info/<location>')
def info(lot_id):
    lot_location = location
    #   Create connection session between SQLAlchemy database and server
    #   Select records in table Spot based on LOT_ID parameter
    #   Store queries in data and push to INFO template
    data = db.session.execute("SELECT * FROM Spot WHERE location = :lot_location;", {"lot_location": lot_location}).fetchall()
    return render_template('info.html', data = data)

@application.route('/show_frame', methods=['GET'])
def show_frame():
    #   test_input = test
    #   var = input('Enter ID: ')
    #   print(FEEDS[var])
    #   frame = FEEDS[var].package(FEEDS[var].norm_frame(FEEDS[var].get_frame()))
    #   print(frame)
    r = requests.get('http://127.0.0.1:8080/get_frame').content
    frame = json.loads(r)
    frame = np.asarray(frame, np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    r, jpg = cv2.imencode('.jpg', frame)
    return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

@application.route('/test', methods=['GET', 'POST'])
def test_2():
    global inc
    stuff = {'id': 0, 'points': []}
    #if request.method == 'POST':
    arr = []
    data = request.json
    #info['points'] = request.json
    if data != None:
        #new_data = len(yaml_loader())
        print(data[0])
        stuff['points'] = [list(data[0]), list(data[1]), list(data[2]), list(data[3])]
        stuff['id'] = inc
        inc = inc + 1
        arr.append(stuff)
        #obj = json.loads(data)
    #print(data)
        with open('config.yml','a') as yamlfile:
            yaml.dump(arr, yamlfile)
    #cur_yaml = yaml.load(yamlfile)
    #cur_yaml.extend(data)
    #print(cur_yaml)

    #with open('config.yml','w') as yamlfile:
    #yaml.dump(data, yamlfile, default_flow_style=False)
                       
    #return render_template("test.html")
    return render_template("test.html")

def yaml_loader():
    with open('config.yml', "r") as yamlfile:
        data2 = yaml.load(yamlfile)
        return data2

# Accept input for Camera Object ID and Address
# Timed JSON call (Loop Image Processing Software)

# Call Process_Frame
# Begin image processing as seperate thread
# When requested to pull frames, get ID and that calls requests

if __name__ == '__main__':
    application.run(host = '127.0.0.1', port = '8090', debug = False)
