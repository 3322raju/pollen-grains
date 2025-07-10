import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from keras.models import load_model
from werkzeug.utils import secure_filename

# Load the trained model
model = load_model("model/pollen_cnn_model.h5", compile=False)

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

# Upload form page
@app.route('/upload.html')
def upload():
    return render_template('upload.html')

# Handle prediction
@app.route('/result', methods=["POST"])
def res():
    if 'file' not in request.files:
        return render_template('upload.html', prediction='No file part')

    f = request.files['file']
    if f.filename == '':
        return render_template('upload.html', prediction='No file selected')

    basepath = os.path.dirname(__file__)
    upload_dir = os.path.join(basepath, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    filename = secure_filename(f.filename)
    filepath = os.path.join(upload_dir, filename)
    f.save(filepath)

    # Load and preprocess image
    img = tf.keras.utils.load_img(filepath, target_size=(150, 150))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Predict
    pred = np.argmax(model.predict(x)[0])
    labels = ['CATTAIL', 'NOT POLLEN', 'NUTSEDGE', 'PARAGRASS']
    result = labels[pred]

    return render_template('upload.html', prediction=result, image_filename=filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
