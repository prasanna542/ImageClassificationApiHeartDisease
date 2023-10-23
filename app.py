# from flask import Flask, request, jsonify
# import tensorflow as tf
# from tensorflow import keras

# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # model = load_model('./model/best_model.h5')
# model = keras.models.load_model('./model/best_model.h5')

# load('./model/logreg.pkl')


from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras

app = Flask(__name__)

# Load the pre-trained machine learning model
model = keras.models.load_model('./model/model_vgg19.h5')

@app.route('/')
def index():
    return('<h1> JSON API for predicting Coronary Heart Disease in a patient. </h1>'+
            '<h2> An example query is <a href="https://heartapi.herokuapp.com/predict?age=31&sex=1&cigs=5&chol=230&sBP=280&dia=0&dBP=90&gluc=87&hRate=84">this</a></h2>'+
            '<h3>It uses parameters such as Age, Sex, Blood Pressure, Diabetes, Glucose and Heart rate for prediction. </h3>'+
            '<h3> This api uses scikit-learn for training the ML model and Flask for serving the webservice. It is trained on a Logistic Regression model with ~88% accuracy</h3>'+
            '<p>For more details view <a href="https://github.com/agoel00/HeartDiseasePredictionAPI">source</a></p>' +
            '<p>Made with 💙 by <a href="https://github.com/agoel00">Anmol Goel</a></p>')




@app.route('/predict', methods=['POST'])
def process_image():
    file = request.files['file']
    img = Image.open(file)
    
    # Convert the image to RGB format
    img = img.convert('RGB')

    # Preprocess the image for your model
    img = img.resize((224, 224))  # Resize to match the model's input size
    img = np.array(img)  # Convert to NumPy array
    img = img / 255.0  # Normalize pixel values

    # Make predictions with the model
    predictions = model.predict(np.expand_dims(img, axis=0))
    return jsonify({'msg': 'success', 'predictions': predictions.tolist()})


if __name__ == "__main__":
    app.run(debug=True)

    
