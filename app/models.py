from app import db, basic_auth
from flask_admin.contrib import sqla
from werkzeug.exceptions import HTTPException

class Lot(db.Model): #Create LOTS table
    id = db.Column(db.Integer, primary_key=True) #Create ID column as PRIMARY KEY
    title = db.Column(db.String(64), index=True) #Create title column
    percentage = db.Column(db.Integer, index=True) #Create percentage column
    total_spaces = db.Column(db.Integer, index=True) #Create total spaces column
    available_spaces = db.Column(db.Integer, index=True) #Create available spaces column
    spots = db.relationship('Spot', backref='author', lazy='dynamic') #Create spots column

    def __repr__(self):
        return '<Lot {}>'.format(self.title)  

class Spot(db.Model): #Create SPOTS table
    id = db.Column(db.Integer, primary_key=True) #Create ID column as PRIMARY KEY
    availability = db.Column(db.Boolean(140)) #Create availability column as boolean
    lot_id = db.Column(db.Integer, db.ForeignKey('lot.id')) #Create lot_id column as FOREIGN KEY

    def __repr__(self):
        return '<Spot {}>'.format(self.availability)

class ModelView(sqla.ModelView):
    def is_accessible(self):
        if not basic_auth.authenticate():
            raise AuthException('Not authenticated.')
        else:
            return True

    def inaccessible_callback(self, name, **kwargs):
        return redirect(basic_auth.challenge())

class AuthException(HTTPException):
    def __init__(self, message):
        super().__init__(message, Response(
            "You could not be authenticated. Please refresh the page.", 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'}
        ))

