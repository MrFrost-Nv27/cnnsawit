from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from . import db
import numpy as np
import pathlib
dataset_path = pathlib.Path("flaskr/static/img/dataset")


api = Blueprint("api", __name__)


@api.route("/dataset", methods=["GET"])
def dataset():
    images = list(filter(lambda item: item.is_file(), dataset_path.rglob("*.jpg")))
    urls = {}

    for image in images:
        name = image.as_posix().replace(dataset_path.as_posix() + "/", "")
        if name.split("/")[0] not in urls:
            urls[name.split("/")[0]] = [name]
        else:
            urls[name.split("/")[0]].append(name)
    return urls, 200

@api.route("/pelatihan/<id>", methods=["GET", "POST", "DELETE"])
def pelatihanbyid(id):
    # data = Models.query.get(id)
    # if request.method == 'POST':
    #     data.nama = request.form.get("nama")
    #     data.algoritma = request.form.get("algoritma")
    #     data.kfold = request.form.get("kfold")
    #     db.session.commit()
    #     return {"toast": {
    #         "icon": "success",
    #         "title": "Data berhasil disimpan"
    #     }, "data": data.serialize()}, 200
    # if request.method == 'DELETE':
    #     db.session.delete(data)
    #     db.session.commit()
    #     return {"toast": {
    #         "icon": "success",
    #         "title": "Data berhasil dihapus"
    #     }}, 200
    # if data == None:
    #     return {"toast": {
    #         "icon": "error",
    #         "title": "Data tidak ditemukan"
    #     }}, 404
    # return {"data": data.serialize()}, 200
    return {"data": ""}, 200