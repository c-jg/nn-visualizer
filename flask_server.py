import json
import tensorflow as tf
import numpy as np

from flask import Flask, request

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
feature_model = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

_, (test_data, _) = tf.keras.datasets.mnist.load_data()
test_data = test_data / 255.0

def get_prediction():
    index = np.random.choice(test_data.shape[0])
    image = test_data[index, :, :]
    image_arr = np.reshape(image, (1,784))
    return feature_model.predict(image_arr), image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        preds, image = get_prediction()
        final_preds = [p.tolist() for p in preds]
        return json.dumps({
            'prediction': final_preds,
            'image': image.tolist()
        })
    return "Welcome to the server."

if __name__ == '__main__':
    app.run()