# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
import itertools

from mpmath import *
from sympy import *

from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from pysymbolic.models.special_functions import MeijerG
from pysymbolic.utilities.performance import compute_Rsquared

from pysymbolic.benchmarks.synthetic_datasets import *
from pysymbolic.utilities.instancewise_metrics import *
from pysymbolic.algorithms.keras_predictive_models import *
from pysymbolic.algorithms.symbolic_expressions import *
from pysymbolic.algorithms.symbolic_metamodeling import *

from sympy.printing.theanocode import theano_function
from sympy.utilities.autowrap import ufuncify

from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from sklearn.neural_network import MLPClassifier

from lime.lime_tabular import LimeTabularExplainer
import shap

import tensorflow as tf
from collections import defaultdict
import re 
import sys
import os
import time
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


class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables. 
    """
    def __init__(self, tau0, k, **kwargs): 
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):   
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)# [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        #random_uniform vs. random.uniform
        uniform = tf.random.uniform(shape =(batch_size, self.k, d), 
            minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
            maxval = 1.0)

        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = K.sigmoid(noisy_logits)
        samples = K.max(samples, axis = 1) 

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
        
        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape 

"""
The code for L2X is adapted from: 
https://github.com/Jianbo-Lab/L2X/blob/master/synthetic/explain.py
Changed further to be more flexible
L2X_flex takes in the training and validation data, while also returning the trained model, meaning we should be able to attack it
"""

def L2X_flex(x_train, y_train, x_val, y_val, activation, num_selected_features, out_activation='sigmoid', 
        loss_='binary_crossentropy', optimizer_='adam', num_hidden=200, num_layers=2, train = True): 
    
    BATCH_SIZE  = len(x_train)
    # x_train, y_train, x_val, y_val, datatype_val = create_data(datatype, n = num_samples)
    input_shape = x_train.shape[1]
     
    # activation = 'relu' if datatype in ['orange_skin','XOR'] else 'selu'
    
    # P(S|X): conditional distribution over P_k
    model_input = Input(shape=(input_shape,), dtype='float32') 

    net = Dense(num_hidden, activation=activation, name = 's/dense1', 
                kernel_regularizer=regularizers.l2(1e-3))(model_input)
    
    
    for _ in range(num_layers-1):
        
        net = Dense(num_hidden, activation=activation, name = 's/dense'+str(_+2), 
                    kernel_regularizer=regularizers.l2(1e-3))(net) 

    # A tensor of shape, [batch_size, max_sents, 100]
    logits = Dense(input_shape)(net) 
    
    # [BATCH_SIZE, max_sents, 1]  
    k = num_selected_features; tau = 0.1
    samples = Sample_Concrete(tau, k, name = 'sample')(logits)

    # q(X_S)
    new_model_input = Multiply()([model_input, samples]) 
    net = Dense(num_hidden, activation=activation, name = 'dense1', 
                kernel_regularizer=regularizers.l2(1e-3))(new_model_input) 
    
    net = BatchNormalization()(net) # Add batchnorm for stability.
    net = Dense(num_hidden, activation=activation, name = 'dense2', 
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = BatchNormalization()(net)

    preds = Dense(1, activation=out_activation, name = 'dense4', 
                  kernel_regularizer=regularizers.l2(1e-3))(net) 
    model = Model(model_input, preds)

    if train: 
        adam = optimizers.Adam(lr = 1e-3)
        model.compile(loss=loss_, 
                      optimizer=optimizer_, 
                      metrics=['acc']) 
        filepath="L2X.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                                     verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, verbose=0, callbacks = callbacks_list, epochs=50, batch_size=BATCH_SIZE) #validation_data=(x_val, y_val)
         
    else:
        model.load_weights('L2X.hdf5', by_name=True) 


    pred_model = Model(model_input, samples)
    pred_model.compile(loss=None, 
                       optimizer='rmsprop',
                       metrics=None)  #metrics=[None]) 
    
    scores = pred_model.predict(x_val, verbose = 0, batch_size = BATCH_SIZE)
    
    ranks = create_rank(scores, num_selected_features)

    median_ranks = compute_median_rank(scores, k = num_selected_features, datatype_val=None)

    return model, pred_model, scores, ranks, median_ranks



