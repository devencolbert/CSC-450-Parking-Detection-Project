import os

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

class Config(object):
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(PROJECT_ROOT, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = False
    SECRET_KEY = 'development-key'
