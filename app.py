from flask import Flask, render_template, url_for, request, send_file, jsonify
import pickle

# Importing the libraries
import ModelTraining as mt
import ModelPrediction as mp
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from matplotlib import pyplot
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Libraries for Creating a model
import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras import regularizers

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/model_training', methods=['POST'])
def train():
    parameters_dictionary = mt.model_training()
    return_json = json.dumps(parameters_dictionary)
    return return_json
	

@app.route('/predict', methods=['POST'])
def predict():
    parameters_dictionary = mp.model_prediction()
    return_json = json.dumps(parameters_dictionary)
    return return_json


@app.route('/download', methods=['POST'])
def download():
    return send_file('Predicted_data.csv', mimetype='text/csv', attachment_filename='Predicted_data.csv',
                     as_attachment=True)


if __name__ == '__main__':
    # app.run(debug=False)
    app.run(debug=False, threaded=False)