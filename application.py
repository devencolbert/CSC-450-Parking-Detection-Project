from flask import Flask, Response, request, render_template, session, json, url_for, abort
import requests
from storage.database.models import *
from flask_sqlalchemy import SQLAlchemy
from flask_admin.contrib.sqla import ModelView
from flask_admin import Admin, BaseView, expose
import os

application = Flask(__name__)

application.config.from_object(__name__)
application.config.update(dict(
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(application.root_path + "/storage/database", "app.db"),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    DEBUG = False,
    SECRET_KEY = 'development-key',
    FLASK_ADMIN_SWATCH = 'yeti'
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

# Take input for ID and Addr
# Timed JSON call (IPS Loop)

# json.loads
# numpy.array
# cv2.imdecode

# Call IPS call process_frame

# Start image process as seperate thread
#    When requested to pull frames, get ID and that calls requests
if __name__ == '__main__':
    application.run(host = '127.0.0.1', port = '8080', debug = True)

