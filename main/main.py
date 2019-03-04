from flask import Flask, render_template, request
#from flask_sqlalchemy import SQLAlchemy
#import requests

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sqlalchemy_example.db'
#db = SQLAlchemy(app)

'''
class Parking(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        position = db.Column(db.String(10), unique=True, nullable=False)
        availability = db.Column(db.Integer(10), unique=True, nullable=False)
        
        def __repr__(self):
               return '<Parking %r>' % self.username
'''


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/view')
def view():
    return render_template('view.html')

@app.route('/info/')
def about():
    return render_template('info.html')