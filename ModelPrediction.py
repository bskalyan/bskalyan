from flask import Flask, render_template, url_for, request, send_file
import pickle

# Importing the libraries
import utilities as util
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


def model_prediction():
    clf = tf.keras.models.load_model('model.h5')
    if request.method == 'POST':
        parameters_dictionary={}
        
        CrossChk, pred_data = util.data_preprocessing("predict")
        
        y_probs = clf.predict(CrossChk, verbose=0)
        # predict crisp classes for test set
        y_classes = clf.predict_classes(CrossChk, verbose=0)
        parameters_dictionary["Message"] = "Model Prediction Completed Successfully"
        # reduce to 1d array
        y_probs = y_probs[:, 0]
        y_prediction = y_classes[:, 0]
        Y_actual = pred_data['y_label']
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(Y_actual, y_prediction)
        print('Accuracy: %f' % accuracy)
        parameters_dictionary["Accuracy"]=accuracy
        # precision tp / (tp + fp)
        precision = precision_score(Y_actual, y_prediction)
        parameters_dictionary["Precision"]=precision
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(Y_actual, y_prediction)
        parameters_dictionary["Recall"]=recall
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(Y_actual, y_prediction)
        parameters_dictionary["F1 score"]=f1
        print('F1 score: %f' % f1)
        pred_data['y_prediction'] = y_prediction
        pred_data['y_probs'] = y_probs
        pred_data.to_csv('Predicted_data.csv', index=False)

    return parameters_dictionary
