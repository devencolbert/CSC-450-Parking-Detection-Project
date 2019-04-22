from app import app, db
from flask import Flask, Response, request, render_template, session
from camera_client import Camera
import yaml

app.secret_key = app.config['SECRET_KEY']

@app.route('/')
@app.route('/index')
def index():
    #Create connection session between SQLAlchemy database and server
    #Select ALL records in tables Lot
    #Store queries in data and push to INDEX template
    data = db.session.execute("SELECT * FROM Lot").fetchall()
    return render_template('index.html', data = data)

@app.route('/info/<location>')
def info(lot_id):
    lot_location = location
    #Create connection session between SQLAlchemy database and server
    #Select records in table Spot based on LOT_ID parameter
    #Store queries in data and push to INFO template
    data = db.session.execute("SELECT * FROM Spot WHERE location = :lot_location;", {"lot_location": lot_location}).fetchall()
    return render_template('info.html', data = data)

@app.route('/test')
def gen(camera_client):
    #while True:
    frame = camera_client.get_frame()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = '8000', debug = True)
