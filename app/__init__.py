from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_admin.contrib.sqla import ModelView
from flask_admin import Admin
from flask_basicauth import BasicAuth

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = 'admin'

basic_auth = BasicAuth(app)

from app import routes, models

admin = Admin(app)
admin.add_view(ModelView(models.Lot, db.session))
admin.add_view(ModelView(models.Spot, db.session))
