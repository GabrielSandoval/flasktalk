from flask import Flask
from flask import request

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import json

app = Flask(__name__)

# Define neural network architecture
def neural_network_model(input_size, lr):
    network = input_data(shape=[None, input_size], name='state')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='iris')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

# Humanize by translating labels
# 0 -> setosa
# 1 -> versicolor
# 2 -> virginica
def humanize_iris_prediction(index):
    if index == 0:
        return 'setosa'
    elif index == 1:
        return 'versicolor'
    elif index == 2:
        return 'virginica'

# Load iris model
model_iris = neural_network_model(4, 0.00001)
model_iris.load('iris_model', weights_only=True)

# Define route
@app.route('/api/predict', methods=['GET'])
def predict_iris():
    # Get params with 'features' key
    # Example:
    # By sending GET request via...
    #   /api/predict?features=[[0.1,4.2,0.3,1.4],[1.2,2.3,3.1,4.2],[0.1,4.2,0.3,1.4]]
    # You'll get the following...
    #   features = [[0.1,4.2,0.3,1.4],[1.2,2.3,3.1,4.2],[0.1,4.2,0.3,1.4]]
    features = json.loads(request.args.get('features'))

    # Predict values
    # Outputs activation values of output nodes.
    # Ex: [[0.96, 0.03, 0.02], [0.02, 0.93, 0.05], [0.06, 0.92, 0.02]]
    predictions = model_iris.predict(features)

    # Outputs humanized predictions.
    # Ex: ['setosa', 'versicolor', 'versicolor', ...]
    predictions = list(map(lambda x: humanize_iris_prediction(x.argmax()), predictions))

    # Return a response containing humanized predictions
    return "{}".format(predictions)

if __name__ == '__main__':
    app.run(debug=True)
