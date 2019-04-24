from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Lot(db.Model):
    id = db.Column(db.Integer, primary_key=True) 
    location = db.Column(db.String(64), index=True)
    
    frames = db.relationship('Frame', backref='Location', lazy='dynamic')
    yamls = db.relationship('Yaml', backref='Location', lazy='dynamic')
    spots = db.relationship('Spot', backref='Location', lazy='dynamic')
    calculations = db.relationship('Calculation', backref='Location', lazy='dynamic')

    def __repr__(self):
        return '<Lot {}>'.format(self.location) 

class Frame(db.Model):
    id = db.Column(db.Integer, primary_key=True) 
    frame_fname = db.Column(db.String(64), index=True) 
    video_fname = db.Column(db.String(64), index=True)  
    lot_location = db.Column(db.String(64), db.ForeignKey('lot.location'))

    def __repr__(self):
        return '<Frame {}>'.format(self.id) 

class Yaml(db.Model):
    id = db.Column(db.Integer, primary_key=True) 
    yaml_fname = db.Column(db.String(64), index=True)  
    lot_location = db.Column(db.String(64), db.ForeignKey('lot.location'))

    def __repr__(self):
        return '<Yaml {}>'.format(self.id)  

class Spot(db.Model): 
    id = db.Column(db.Integer, primary_key=True)  
    spot_id = db.Column(db.Integer, index=True)
    availability = db.Column(db.Boolean(140), index=True)
    lot_location = db.Column(db.String(64), db.ForeignKey('lot.location'))

    def __repr__(self):
        return '<Spot {}>'.format(self.id)

class Calculation(db.Model): 
    id = db.Column(db.Integer, primary_key=True)  
    total_spots = db.Column(db.Integer, index=True)
    available_spots = db.Column(db.Integer, index=True)
    percentage = db.Column(db.Integer, index=True)
    lot_location = db.Column(db.String(64), db.ForeignKey('lot.location'))

    def __repr__(self):
        return '<Calculations {}>'.format(self.id)


