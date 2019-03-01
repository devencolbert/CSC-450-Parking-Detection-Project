from app import db

class Lot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(64), index=True)
    spots = db.relationship('Spot', backref='author', lazy='dynamic')

    def __repr__(self):
        return '<Lot {}>'.format(self.username)  

class Spot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    availability = db.Column(db.String(140))
    lot_id = db.Column(db.Integer, db.ForeignKey('lot.id'))

    def __repr__(self):
        return '<Spot {}>'.format(self.body)
