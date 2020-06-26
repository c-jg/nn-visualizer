import streamlit as st
import json, requests
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np


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

def pre():
    preds, image = get_prediction()
    final_preds = [p.tolist() for p in preds]
    return json.dumps({
        'prediction': final_preds,
        'image': image.tolist()
    })

st.title('Neural Network Visualizer')
st.markdown('## Input Image:')

if st.button('Get Random Prediction'):
    p = pre()
    response = json.loads(p)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28,28))

    st.image(image, width=150)

    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(32,4))

        if layer == 2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16
        
        for i, number in enumerate(numbers):
            plt.subplot(row, col, i+1)
            plt.imshow(number * np.ones((8,8,3)).astype('float32'))
            plt.xticks([])
            plt.yticks([])

            if layer == 2:
                plt.xlabel(str(i), fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text(f"Layer {layer + 1}")
        st.pyplot()