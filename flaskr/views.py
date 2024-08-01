from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from . import db

views = Blueprint("views", __name__)


@views.route("/")
def index():
    return render_template(
        "page/index.html",
        page="home",
    )

@views.route("/home")
def home():
    return render_template(
        "page/index.html",
        page="home",
    )

@views.route("/pelatihan")
def pelatihan():
    return render_template(
        "page/pelatihan.html",
        page="pelatihan",
    )

@views.route("/prediksi")
def prediksi():
    return render_template(
        "page/prediksi.html",
        page="prediksi",
    )