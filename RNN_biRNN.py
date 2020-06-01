from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# model = tf.keras.Sequential()
# model.add(layers.Embedding(input_dim=1000, output_dim=64))
# model.add(layers.LSTM(128, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.summary()

# model = tf.keras.Sequential()
# model.add(layers.Embedding(input_dim=1000, output_dim=64))
# model.add(layers.GRU(256, return_sequences=True))
# model.add(layers.SimpleRNN(128))
# model.add(layers.Dense(10, activation='softmax'))
# model.summary()

# encoder_vocab = 1000
# decoder_vocab = 2000

# encoder_input = layers.Input(shape=(None, ))
# encoder_embedded = layers.Embedding(input_dim=encoder_vocab,
#                                     output_dim=64)(encoder_input)
# output,state_h,state_c = layers.LSTM(
#   64,return_state=True,name='encoder')(encoder_embedded)
# encoder_state = [state_h,state_c]

# decoder_input = layers.Input(shape=(None,))
# decoder_embedded=layers.Embedding(input_dim=decoder_vocab,output_dim=64)(decoder_input)

# decoder_output = layers.LSTM(
#   64,name='decoder')(decoder_embedded,initial_state=encoder_state)
# output=layers.Dense(10,activation='softmax')(decoder_output)

# model = tf.keras.Model([encoder_input,decoder_input],output)
# model.summary()




