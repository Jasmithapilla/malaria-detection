import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

model = load_model("model/malaria_model_cnn.h5")

IMG_SIZE = 224

def predict_image(img_path):
    img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "Parasitized (Malaria Infected)"
    else:
        return "Uninfected (Healthy Cell)"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    result = predict_image(filepath)

    return render_template("result.html", prediction=result, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)