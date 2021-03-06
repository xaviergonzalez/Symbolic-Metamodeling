#standard libraries
import numpy as np
import matplotlib.pyplot as plt
#explanatory libraries from research papers
from pysymbolic.benchmarks.synthetic_datasets import *
from pysymbolic.algorithms.keras_predictive_models import *
from pysymbolic.algorithms.instancewise_feature_selection import *
from pysymbolic.algorithms.L2X import *
#keras
from tensorflow import keras
from keras.callbacks import ModelCheckpoint    
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer 
import json
import random
from keras import optimizers
#adversarial example library
import cleverhans
from cleverhans.future.tf2.attacks import fast_gradient_method
#for creating file path to save checkpoints
import os
import datetime
from datetime import date
#make warnings quiet
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
#for printing out training results
from pysymbolic.algorithms.record import *

#creates model
def compile_model(input_shape, num_hidden = 200, regularize = False):
    if regularize:
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden, input_shape=(input_shape,), kernel_regularizer=regularizers.l2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2),
        tf.keras.layers.Activation(tf.nn.softmax)]) #separate activation to get logits
        adam = optimizers.Adam(lr = 1e-3)
    else:
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden, input_shape=(input_shape,)),
        tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu),
        tf.keras.layers.Dense(2),
        tf.keras.layers.Activation(tf.nn.softmax)]) #separate activation to get logits
    model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy', #should be 'sparse_categorical_crossentropy' b/c one-hot encoded
              metrics=['accuracy'])
    logits_model = tf.keras.Model(model.input,model.layers[-1].output)
#     print(model.summary) #how could I get this actually to print the model summary?
    return model, logits_model
SOFT_MESSAGE = """
def XOR_model(input_shape, num_hidden = 200):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_hidden, input_shape=(input_shape,), kernel_regularizer=regularizers.l2(1e-3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(1e-3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l2(1e-3)),
    tf.keras.layers.Activation(tf.nn.softmax)]) #separate activation to get logits
    adam = optimizers.Adam(lr = 1e-3)
    model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
    logits_model = tf.keras.Model(model.input,model.layers[-1].output)
    return model, logits_model
"""

def compile_model_sig(input_shape, num_hidden = 200, regularize = False):
    if regularize:
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden, input_shape=(input_shape,), kernel_regularizer=regularizers.l2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, kernel_regularizer=regularizers.l2(1e-3)),
        tf.keras.layers.Activation(tf.nn.sigmoid)]) #separate activation to get logits
        adam = optimizers.Adam(lr = 1e-3)
    else:
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden, input_shape=(input_shape,)),
        tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, kernel_regularizer=regularizers.l2(1e-3)),
        tf.keras.layers.Activation(tf.nn.sigmoid)]) #separate activation to get logits
    model.compile(optimizer='adam',
              loss= 'binary_crossentropy', #should be binary_crossentropy b/c only one-output
              metrics=['accuracy'])
    logits_model = tf.keras.Model(model.input,model.layers[-1].output)
#     print(model.summary) #how could I get this actually to print the model summary?
    return model, logits_model
SIG_MESSAGE = '''def XOR_model_mimic_l2x(input_shape, num_hidden = 200):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_hidden, input_shape=(input_shape,), kernel_regularizer=regularizers.l2(1e-3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(1e-3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, kernel_regularizer=regularizers.l2(1e-3)),
    tf.keras.layers.Activation(tf.nn.sigmoid)]) #separate activation to get logits
    adam = optimizers.Adam(lr = 1e-3)
    model.compile(optimizer='adam',
              loss= 'binary_crossentropy',
              metrics=['accuracy'])
    logits_model = tf.keras.Model(model.input,model.layers[-1].output)
    return model, logits_model'''

#save model controls whether or not we pass callbacks, need a save_dir
#how do I save the number of epochs?
#callbacks needs to be a list
def train_model(model, x_train, y_train, x_val, y_val, save_model, save_dir, callbacks, 
                    message, notebook = "tst", epochs = 1, verbose = 1):
    if save_model:
        history = model.fit(x_train, 
              y_train, 
              epochs = epochs, 
              validation_data = (x_val, y_val), 
              callbacks = callbacks,
              verbose = verbose) 
        write_metadata(save_dir, notebook, epochs, len(x_train), message)
    else:
        history = model.fit(x_train, 
                            y_train, 
                            epochs = epochs, 
                            validation_data = (x_val, y_val),
                            verbose = verbose)
    graph_loss(history, save_dir, save_model)
    return history

#returns model for classifying XOR synthetic data, along with directory for saving
#activation is either "soft" for softmax or "sig" for sigmoid
def build_model(feats, num_hidden, name, activation, regularize = False, verbose = 0):
    DATE = date.today()
    if activation == "soft":
        model, logits_model = compile_model(feats, num_hidden = num_hidden, regularize = regularize)
    else:
        model, logits_model = compile_model_sig(feats, num_hidden = num_hidden, regularize = regularize)
    checkpoint_path = str(DATE) + activation + name + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=verbose)
    return model, logits_model, checkpoint_path, checkpoint_dir, cp_callback

#returns list indicating which examples classified correctly (true) or incorrectly (false)
#activation is string: "soft" for softmax activation, "sig" for sigmoid activation
def misclass_lst(x_adv, y_adv, model, activation):
    if activation == "soft":
        adv_class_lst = list(map(lambda x: list(x).index(max(x)), model.predict(x_adv))) == y_adv
    if activation == 'sig':
        adv_class_lst = [int(round(x[0])) for x in model.predict(x_adv)] == y_adv
    return adv_class_lst

#returns indices of which examples are misclassified
def misclass_index(x_adv, y_adv, model, activation):
    class_lst = misclass_lst(x_adv, y_adv, model, activation)
    return [i for i, x in enumerate(class_lst) if Not(x)]

#split indices of misclassified examples into substantial and insubstantial misclassification
def error_breakdown(datatype, x_adv,y_adv, model, activation):
    errors = misclass_index(x_adv,y_adv, model, activation)
    sub_errors = []
    insub_errors = []
    datatype_dict  = {"XOR": xor_thresh_fxn, "orange_skin": orange_thresh_fxn, "nonlinear_additive" :additive_thresh_fxn, "switch": switch_thresh_fxn}
    thresh_fxn = datatype_dict[datatype]
    for i in errors:
        if thresh_fxn(x_adv, i):
            sub_errors.append(i)
        else:
            insub_errors.append(i)
    return errors, sub_errors, insub_errors

def xor_thresh_fxn(arr, i):
    return abs(arr[i,0] * arr[i,1]) > 0.1

def orange_thresh_fxn(arr, i):
    return abs(np.sum(arr[i,:4]**2) - 4) > 0.1

def additive_thresh_fxn(arr, i):
    return abs(-100 * np.sin(0.2*arr[i,0]) + abs(arr[i,1]) + arr[i,2] + np.exp(-arr[i,3])  - 2.4) > 0.1

def switch_thresh_fxn(arr, i):
    if abs(arr[i, -1]) < 0.1:
        return False
    elif arr[i, -1] > 0.1:
        return orange_thresh_fxn(arr,i)
    else:
        return additive_thresh_fxn(arr[:, 4:], i)

#for given model and validation set, analyzes whether errors are "significant" or not (based on how the data was created)
def print_error_breakdown(datatype, x_val, y_val, model, activation, mod_name):
    error, sub_error, insub_error = error_breakdown(datatype, x_val, y_val, model, activation)
    denom = len(y_val)
    print(mod_name)
    print("percent of errors: ", 100 * len(error) / denom)
    print("percent of insubstantial errors: ", 100 * len(insub_error) / denom)
    print("percent of substantial errors: ", 100 * len(sub_error) / denom)
    print()
    
def generate_switch_labels(X):
    y = (X[:, -1] > 0) * (np.sum(X[:,:4]**2, axis = 1) - 4.0) + (1 - (X[:, -1] > 0)) * (-100 * np.sin(0.2*X[:,4]) + abs(X[:,5]) + X[:,6] + np.exp(-X[:,7])  - 2.4)
    y = (y > 0) * 1
    return y
    
def test_for_sub_error(name, datatype, feats = 10, n_train = 10 ** 4, n_val = 10 ** 4, epochs = 10, epsilon = 0.3, verbose = 0, save_model = True):
    #prepare the training data
    num_hidden = 200
    datatype_dict = {"XOR": 2, "orange_skin":4, "nonlinear_additive" : 4, "switch" : 5}
    x_train, y_train, x_val, y_val, _ = create_data(datatype, n = n_train, nval = n_val, feats = feats)
    #initialize and train the various models
    soft_mod, soft_logits_mod, soft_path, soft_dir, soft_cp = build_model(feats, num_hidden, name, "soft")
    train_model(soft_mod, x_train, y_train, x_val, y_val, save_model, soft_dir, [soft_cp], SOFT_MESSAGE, epochs = epochs, verbose = verbose)
    sig_mod, sig_logits_mod, sig_path, sig_dir, sig_cp = build_model(feats, num_hidden, name, "sig")
    train_model(sig_mod, x_train, y_train, x_val, y_val, save_model, sig_dir, [sig_cp], SIG_MESSAGE, epochs = epochs, verbose = verbose)
    l2x_mod, l2x_logit_mod, l2x_pred_mod, _, _, _ = L2X_flex(x_train, y_train, x_val, y_val, activation = 'relu', filedir = str(date.today()) + "l2x" + 
                                                  name,num_selected_features = datatype_dict[datatype], out_activation='sigmoid', 
                                                  loss='binary_crossentropy', optimizer='adam', num_hidden=num_hidden, num_layers=2, 
                                                  train = True, epochs = epochs, verbose = verbose)
    #create the adversarial examples
    epsilon = epsilon
    x_adv = fast_gradient_method(soft_logits_mod, x_val, epsilon, np.inf, targeted=False)
    x_adv = x_adv.numpy() #turn to np.array from tf object
    #create correct labels for y_adv
    if datatype == "XOR":
        y_adv = generate_XOR_labels(x_adv)
    elif datatype == "orange_skin":
        y_adv = generate_orange_labels(x_adv)
    elif datatype == "nonlinear_additive":
        y_adv = generate_additive_labels(x_adv)
    if datatype != "switch":
        y_adv = (y_adv[:,0]>0.5)*1
    else:
        y_adv = generate_switch_labels(x_adv)
    print_error_breakdown(datatype, x_val, y_val, soft_mod, "soft", "soft val " + name + datatype)
    print_error_breakdown(datatype, x_adv, y_adv, soft_mod, "soft", "soft adv " + name + datatype)
    print_error_breakdown(datatype, x_val, y_val, sig_mod, "sig", "sig val " + name + datatype)
    print_error_breakdown(datatype, x_adv, y_adv, sig_mod, "sig", "sig adv " + name + datatype )
    print_error_breakdown(datatype, x_val, y_val, l2x_mod, "sig", "l2x val " + name + datatype )
    print_error_breakdown(datatype, x_adv, y_adv, l2x_mod, "sig", "l2x adv " + name + datatype)
    #return the information needed to see the effect of the adversarial examples on the predictions
    return x_val, y_val, x_adv, y_adv, l2x_mod, l2x_logit_mod, l2x_pred_mod
    
def feats_to_error_l2x(feats, epochs = 1, verbose = 1):
    fd = "throwaway"
    epsilon = 0.3
    x_train, y_train, x_val, y_val, _ = create_data("XOR", n = np.int(1e4), feats = feats)
    x_train = np.float32(x_train)
    y_train = np.int32(y_train)
    x_val = np.float32(x_val)
    y_val = np.int32(y_val)
    l2x_model, _, _, _, _ = L2X_flex(x_train, y_train, x_val, y_val, activation = 'relu', filedir = "throwaway", num_selected_features = 2,
                                     out_activation='sigmoid', 
                                     loss='binary_crossentropy', optimizer='adam', num_hidden=200, num_layers=2, train = True, epochs =
                                     epochs)
    l2x_model.evaluate(x_val, y_val, verbose = verbose)