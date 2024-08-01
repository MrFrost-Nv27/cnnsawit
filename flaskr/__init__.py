import os

from functools import wraps
from flask import Flask, render_template, redirect, flash, url_for, current_app, request
from flask_sqlalchemy import SQLAlchemy
from flaskr.config import Config

db = SQLAlchemy()

def create_app(config_class=Config):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    db.init_app(app)

    from .views import views
    from .api import api
    
    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html"), 404

    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(api, url_prefix="/api")

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    return app
