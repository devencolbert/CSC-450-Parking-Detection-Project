from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_admin.contrib.sqla import ModelView
from flask_admin import Admin, BaseView, expose

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from app import routes, models

app.config['FLASK_ADMIN_SWATCH'] = 'yeti'

admin = Admin(app, name='Admin Portal', template_mode = 'bootstrap3')

class ConfigurationView(BaseView):
    @expose('/')
    def index(self):
        return self.render('configuration.html')

admin.add_view(ConfigurationView(name='Configuration', endpoint='configuration'))
admin.add_view(ModelView(models.Lot, db.session, category="Database"))
admin.add_view(ModelView(models.Spot, db.session, category="Database"))
