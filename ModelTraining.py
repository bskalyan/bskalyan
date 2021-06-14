from flask import Flask, render_template, url_for, request, send_file
import pickle

# Importing the libraries
import utilities as util
import ast
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


def model_training():
    if request.method == 'POST':
        batch_size = int(request.args.get('batch_size'))
        epochs = int(request.args.get('epochs'))
        validation_split = float(request.args.get('validation_split'))
        optimizer = (request.args.get('optimizer'))
        dropout = float(request.args.get('dropout'))
        regularizer = float(request.args.get('regularizer'))
        layers = int(request.args.get('layers'))
        network_nodes = ast.literal_eval(request.args.get('network_nodes'))
		
        parameters_dictionary = {}

        X_train, Y_data = util.data_preprocessing("train")
        
		
        # splitting data to tarin and test with 80:20 rule
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_train, Y_data, test_size=0.2, shuffle=True, stratify=Y_data, random_state=42)

        # Building model network
        model = Sequential()
        model.add(Dense(network_nodes[0], input_dim=7, activation='relu', kernel_constraint=maxnorm(3),
                        kernel_regularizer=regularizers.l2(regularizer)))
        model.add(Dropout(rate=dropout))
        for layer in range(layers-1):
            model.add(Dense(network_nodes[layer+1], activation='relu', kernel_constraint=maxnorm(3), kernel_regularizer=regularizers.l2(regularizer)))
            model.add(Dropout(rate=dropout))
        model.add(Dense(1, activation='sigmoid'))

        # compiling model
        from keras.callbacks import EarlyStopping
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(Xtrain, Ytrain, validation_split=validation_split, batch_size=batch_size, epochs=epochs, verbose=1)
        parameters_dictionary["Message"] = "Model Training Completed Successfully"
        print('****************Model Training is DONE*****************')

        import time
        # estimate accuracy on whole dataset using loaded weights
        scores = model.evaluate(Xtrain, Ytrain, verbose=0)
        start = time.time()
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        end = time.time()
        print("time:", end - start)

        # Model evalauting
        # predict probabilities for test set
        yhat_probs = model.predict(Xtest, verbose=0)
        # predict crisp classes for test set
        yhat_classes = model.predict_classes(Xtest, verbose=0)
        print('****************Model Evaluating is done*********************')

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(Ytest, yhat_classes)
        parameters_dictionary["Accuracy"]=accuracy
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(Ytest, yhat_classes)
        parameters_dictionary["Precision"]=precision
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(Ytest, yhat_classes)
        parameters_dictionary["Recall"]=recall
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(Ytest, yhat_classes)
        parameters_dictionary["F1 score"]=f1
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(Ytest, yhat_classes)
        parameters_dictionary["Cohens Kappa"]=kappa
        print('Cohens kappa: %f' % kappa)
        # ROC AUC
        auc = roc_auc_score(Ytest, yhat_probs)
        parameters_dictionary["ROC AUC"]=auc
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(Ytest, yhat_classes)
        model.save("model.h5")

    return parameters_dictionary