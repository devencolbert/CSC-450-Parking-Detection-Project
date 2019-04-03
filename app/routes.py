from app import app, db
from flask import Flask, request, render_template, session
#import datasets

app.secret_key = app.config['SECRET_KEY']

@app.route('/')
@app.route('/index')
def index():
    #Create connection session between SQLAlchemy database and server
    #Select ALL records in tables Lot
    #Store queries in data and push to INDEX template
    data = db.session.execute("SELECT * FROM Lot").fetchall()
    return render_template('index.html', data=data)

@app.route('/info/<location>')
def info(lot_id):
    lot_location = location
    #Create connection session between SQLAlchemy database and server
    #Select records in table Spot based on LOT_ID parameter
    #Store queries in data and push to INFO template
    data = db.session.execute("SELECT * FROM Spot WHERE location = :lot_location;", {"lot_location": lot_location}).fetchall()
    return render_template('info.html', data=data)

#@app.route('/test')
#def test():
    #return datasets.main()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)
