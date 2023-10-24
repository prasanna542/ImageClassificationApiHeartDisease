from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras

app = Flask(__name__)

# Load the pre-trained machine learning model
model = keras.models.load_model('model_vgg19.h5')

@app.route('/')
def index():
    # return('<h1> JSON API for predicting Coronary Heart Disease in a patient. </h1>')
    return render_template('index.html')



@app.route('/', methods=['POST'])
def process_image():
    file = request.files['file']

    image_path = "./images/"+ file.filename
    file.save(image_path)




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

    
