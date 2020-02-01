import keras
from keras.models import Sequential, Model, load_model
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import keras.losses

import tensorflow as tf

import isolearn.keras as iso

import numpy as np


#DragoNN Saved Model definition

def load_saved_predictor(model_path, library_context=None) :
	
	def dummy_pred_func(y_true, y_pred) :
		return y_pred

	saved_model = load_model(model_path)
	#print(saved_model.summary())

	def _initialize_predictor_weights(predictor_model, saved_model=saved_model) :
		#Load pre-trained model
		predictor_model.get_layer('optimus5_conv1d_1').set_weights(saved_model.get_layer('convolution1d_1').get_weights())
		predictor_model.get_layer('optimus5_conv1d_1').trainable = False
		
		predictor_model.get_layer('optimus5_conv1d_2').set_weights(saved_model.get_layer('convolution1d_2').get_weights())
		predictor_model.get_layer('optimus5_conv1d_2').trainable = False

		predictor_model.get_layer('optimus5_dense_1').set_weights(saved_model.get_layer('dense_1').get_weights())
		predictor_model.get_layer('optimus5_dense_1').trainable = False

		predictor_model.get_layer('optimus5_dense_2').set_weights(saved_model.get_layer('dense_2').get_weights())
		predictor_model.get_layer('optimus5_dense_2').trainable = False

	def _initialize_predictor_weights_old(predictor_model, saved_model=saved_model, model_path=model_path) :
		#Load pre-trained model
		#print(saved_model.summary())
		predictor_model.load_weights(model_path, by_name=True)

	def _load_predictor_func(sequence_input) :
		#DragoNN parameters
		seq_length = 1000
		seq_input_shape = (seq_length, 4, 1)
		n_tasks = 1

		#Define model layers
		permute_input = Lambda(lambda x: x[..., 0])
		
		conv_1 = Conv1D(40, 8, padding='valid', activation='linear', name='optimus5_conv1d_1')
		relu_1 = Activation('relu')
		
		conv_2 = Conv1D(40, 8, padding='valid', activation='linear', name='optimus5_conv1d_2')
		relu_2 = Activation('relu')
		
		flatten_3 = Flatten()
		dense_4 = Dense(40, name='optimus5_dense_1')
		relu_4 = Activation('relu')
		drop_4 = Dropout(0.2)
		
		final_dense = Dense(1, name='optimus5_dense_2')

		#Execute functional model definition
		permuted_input = permute_input(sequence_input)
		
		relu_1_out = relu_1(conv_1(permuted_input))
		relu_2_out = relu_2(conv_2(relu_1_out))

		relu_4_out = drop_4(relu_4(dense_4(flatten_3(relu_2_out))), training=False)
		
		final_dense_out = final_dense(relu_4_out)

		predictor_inputs = []
		predictor_outputs = [final_dense_out]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
