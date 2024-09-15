from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify, send_file
from . import db
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
from re import sub
import os
import matplotlib.pyplot as plt


TF_ENABLE_ONEDNN_OPTS = 0


api = Blueprint("api", __name__)


@api.route("/dataset", methods=["GET"])
def dataset():
    dataset_path = pathlib.Path("flaskr/static/img/dataset")
    images = list(filter(lambda item: item.is_file(),
                  dataset_path.rglob("*.jpg")))
    urls = {}

    for image in images:
        name = image.as_posix().replace(dataset_path.as_posix() + "/", "")
        if name.split("/")[0] not in urls:
            urls[name.split("/")[0]] = [name]
        else:
            urls[name.split("/")[0]].append(name)
    return urls, 200


@api.route("/pelatihan", methods=["POST"])
def pelatihan():
    name = '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
            sub('([A-Z]+)', r' \1',
                str(request.form.get("name")).replace('-', ' '))).split()).lower()

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest')
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'flaskr/static/img/dataset', target_size=(256, 256), batch_size=32, class_mode='categorical')
    valid_generator = valid_datagen.flow_from_directory(
        'flaskr/static/img/dataset', target_size=(256, 256), batch_size=32, class_mode='categorical')

    class_names = list(train_generator.class_indices.keys())
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               input_shape=train_generator.image_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(train_generator.image_shape[0], activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        epochs=5
    )

    if not os.path.exists('models'):
        os.makedirs('models')
    scores = model.evaluate(valid_generator)

    class_labels = list(train_generator.class_indices.keys())

    with open(f'models/{name}.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    with open(f'models/{name}.class', 'wb') as file_pi:
        pickle.dump(class_labels, file_pi)
    with open(f'models/{name}.scores', 'wb') as file_pi:
        pickle.dump(scores, file_pi)
    model.save(f'models/{name}.keras')

    # Plot grafik akurasi
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Simpan grafik ke file
    plt.savefig(f'models/{name}_accuracy_plot.png')  # Simpan dalam format PNG

    plt.close()  # Menutup grafik setelah menyimpannya
    

    return {
        "scores": scores,
        "history": history.history,
        "class_labels": class_labels,
    }


@api.route("/prediksi", methods=["POST"])
def prediksi():
    name = '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
            sub('([A-Z]+)', r' \1',
                str(request.form.get("name")).replace('-', ' '))).split()).lower()
    img = request.files['image']
    img.save(f'models/{img.filename}')

    model = load_model(f'models/{name}.keras')
    with open(f'models/{name}.class', "rb") as file_pi:
        class_labels = pickle.load(file_pi)

    img = image.load_img(f'models/{img.filename}', target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return {
        # "class": predicted_class[0],
        "label": class_labels[predicted_class[0]],
        "predictions": predictions[0].tolist(),
        "classes": class_labels,
    }, 200


@api.route("/models", methods=["GET"])
def models():
    dataset_path = pathlib.Path("models")
    models = list(filter(lambda item: item.is_file(),
                  dataset_path.rglob("*.keras")))
    names = []

    for model in models:
        name = model.as_posix().replace(
            dataset_path.as_posix() + "/", "").replace(".keras", "")

        with open(f'models/{name}.history', "rb") as file_pi:
            history = pickle.load(file_pi)
        with open(f'models/{name}.class', "rb") as file_pi:
            class_labels = pickle.load(file_pi)
        with open(f'models/{name}.scores', "rb") as file_pi:
            scores = pickle.load(file_pi)
        
        names.append({
            "name": name,
            "akurasi": scores[1]*100,
            "history": history,
            "class_labels": class_labels
        })
    return names, 200


@api.route("/models/<name>", methods=["GET", "DELETE"])
def modelsDelete(name):
    if request.method == 'DELETE':
        model_path = pathlib.Path(f"models/{name}.keras")
        history_path = pathlib.Path(f"models/{name}.history")
        class_path = pathlib.Path(f"models/{name}.class")
        scores_path = pathlib.Path(f"models/{name}.scores")

        # Delete the files if they exist
        if model_path.exists():
            model_path.unlink()
        if history_path.exists():
            history_path.unlink()
        if class_path.exists():
            class_path.unlink()
        if scores_path.exists():
            scores_path.unlink()

        return jsonify({"toast": {"icon": "success", "title": "Data berhasil dihapus"}}), 200

    # For GET method, return the model details
    with open(f'models/{name}.history', "rb") as file_pi:
        history = pickle.load(file_pi)
    with open(f'models/{name}.class', "rb") as file_pi:
        class_labels = pickle.load(file_pi)
    with open(f'models/{name}.scores', "rb") as file_pi:
        scores = pickle.load(file_pi)

    return jsonify({
        "history": history,
        "class_labels": class_labels,
        "scores": scores
    })

@api.route("/models/<name>/accuracy_plot", methods=["GET"])
def get_accuracy_plot(name):
    image_path = os.path.join(os.getcwd(), f'models/{name}_accuracy_plot.png')
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return jsonify({"error": "Image not found"}), 404