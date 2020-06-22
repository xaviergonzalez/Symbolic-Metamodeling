#standard libraries
import numpy as np
import matplotlib.pyplot as plt
#keras
import tensorflow as tf
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
#for creating file path to save checkpoints
import os
import datetime
from datetime import date
#make warnings quiet
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#produces graph of accuracy and loss over training time
def graph_loss(history, filedir):
    # summarize history for accuracy metric
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filedir + '/acc.png')
    print("saving graph")
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filedir + '/loss.png')
    plt.show()