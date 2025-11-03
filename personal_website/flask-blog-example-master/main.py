import flask
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import json
from datetime import datetime

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

app = Flask(__name__)
local_server=True
if local_server:
    app.config["SQLALCHEMY_DATABASE_URI"] = params["local_uri"]
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = params["prod_uri"]

db = SQLAlchemy(app)

class Contacts(db.Model):
    sno = db.Column(db.String(80), nullable=True, unique=False)
    Name = db.Column(db.String(80), unique=False, primary_key=True)
    email = db.Column(db.String(80), unique=False)
    phone_number = db.Column(db.Integer(), unique=False)
    Message = db.Column(db.String(200), unique=False)
    time = db.Column(db.String(12), nullable=True, unique=True)


class Posts(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    Title = db.Column(db.String(80), nullable=False, unique=False)
    Subtitle = db.Column(db.String(80), nullable=False, unique=False)
    slug = db.Column(db.String(25), nullable=False, unique=False)
    Author = db.Column(db.String(50), nullable=False, unique=False)
    Content = db.Column(db.String(200), nullable=False, unique=False)
    img_file = db.Column(db.String(25), nullable=False, unique=False)

@app.route("/")
def start():
    posts = Posts.query.filter_by().all()[0:5]
    return render_template('index.html', params=params, posts=posts)


@app.route("/index.html")
def index():
    posts = Posts.query.filter_by().all()[0:5]
    return render_template('index.html', params=params, posts=posts)


@app.route("/post/<string:post_slug>", methods=['GET'])
def post_route(post_slug):
    post = Posts.query.filter_by(slug=post_slug).first()
    return render_template('post.html', params=params, posti=post)


@app.route("/contact.html", methods=['GET', 'POST'])
def contact():
    if(request.method=='POST'):
        '''Add entry to the database'''
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        entry = Contacts(Name=name, phone_number=phone, Message=message, time=datetime.now(), email=email )
        db.session.add(entry)
        db.session.commit()
    return render_template('contact.html', params=params)


@app.route("/about.html")
def about():
    return render_template('about.html',  params=params)


app.run(debug=True)

'''debug = True helps to make dynamic changes in your function which will be reflected on the website without a 
rerun
'''
