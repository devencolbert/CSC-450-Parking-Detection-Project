from app import app, basic_auth
from flask_login import current_user, login_user
from flask import Flask, flash, redirect, request, render_template, session, url_for
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

@app.route('/info/<lot_id>')
def info(lot_id):
    lotid = lot_id
    con = sqlite3.connect(DATABASE)
    cur = con.cursor()
    cur.execute("SELECT * FROM Spot WHERE lot_id = ?", (lotid))
    data = cur.fetchall()
    return render_template('info.html', data=data)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)