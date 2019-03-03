import os

basedir = os.path.abspath(os.path.dirname(__file__))
db_uri = 'sqlite:///{}'.format(basedir)

class Config(object):
    SECRET_KEY = 'in-case-we-decide-to-use-this'
    SQLALCHEMY_DATABASE_URI = db_uri
    SQLALCHEMY_TRACK_MODIFICATIONS = False
