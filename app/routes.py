from app import app
from flask import Flask, request, render_template
import sqlite3

DATABASE = 'C:\\Users\\Connor\\Documents\\project\\app.db'
print(DATABASE) #temporary static directory path

@app.route('/')
@app.route('/index')
def index():
    con = sqlite3.connect(DATABASE)
    cur = con.cursor()
    cur.execute("SELECT * FROM Lot")
    data = cur.fetchall()
    return render_template('index.html', data=data)
