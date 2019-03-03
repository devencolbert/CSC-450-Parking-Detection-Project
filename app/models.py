from app import db

class Lot(db.Model): #Create LOTS table
    id = db.Column(db.Integer, primary_key=True) #Create ID column as PRIMARY KEY
    title = db.Column(db.String(64), index=True) #Create title column
    spots = db.relationship('Spot', backref='author', lazy='dynamic') #Create spots column

    def __repr__(self):
        return '<Lot {}>'.format(self.username)  

class Spot(db.Model): #Create SPOTS table
    id = db.Column(db.Integer, primary_key=True) #Create ID column as PRIMARY KEY
    availability = db.Column(db.Boolean(140)) #Create availability column as boolean
    lot_id = db.Column(db.Integer, db.ForeignKey('lot.id')) #Create lot_id column as FOREIGN KEY

    def __repr__(self):
        return '<Spot {}>'.format(self.body)
