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
	
	saved_model = Sequential()

	saved_model.add(Conv1D(120, 5, activation='relu', input_shape=(145, 4), name='dragonn_conv1d_1_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_1_copy'))
	saved_model.add(Dropout(0.1))

	saved_model.add(Conv1D(120, 5, activation='relu', name='dragonn_conv1d_2_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_2_copy'))
	saved_model.add(Dropout(0.1))

	saved_model.add(Conv1D(120, 5, activation='relu', name='dragonn_conv1d_3_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_3_copy'))
	saved_model.add(Dropout(0.1))

	saved_model.add(Flatten())
	saved_model.add(Dense(12, activation='linear', name='dragonn_dense_1_copy'))

	saved_model.compile(
		loss= "mean_squared_error",
		optimizer=keras.optimizers.SGD(lr=0.1)
	)

	saved_model.load_weights(model_path)
	#print(saved_model.summary())

	def _initialize_predictor_weights(predictor_model, saved_model=saved_model) :
		#Load pre-trained model
		predictor_model.get_layer('dragonn_conv1d_1').set_weights(saved_model.get_layer('dragonn_conv1d_1_copy').get_weights())
		predictor_model.get_layer('dragonn_conv1d_1').trainable = False
		predictor_model.get_layer('dragonn_batchnorm_1').set_weights(saved_model.get_layer('dragonn_batchnorm_1_copy').get_weights())
		predictor_model.get_layer('dragonn_batchnorm_1').trainable = False

		predictor_model.get_layer('dragonn_conv1d_2').set_weights(saved_model.get_layer('dragonn_conv1d_2_copy').get_weights())
		predictor_model.get_layer('dragonn_conv1d_2').trainable = False
		predictor_model.get_layer('dragonn_batchnorm_2').set_weights(saved_model.get_layer('dragonn_batchnorm_2_copy').get_weights())
		predictor_model.get_layer('dragonn_batchnorm_2').trainable = False
		
		predictor_model.get_layer('dragonn_conv1d_3').set_weights(saved_model.get_layer('dragonn_conv1d_3_copy').get_weights())
		predictor_model.get_layer('dragonn_conv1d_3').trainable = False
		predictor_model.get_layer('dragonn_batchnorm_3').set_weights(saved_model.get_layer('dragonn_batchnorm_3_copy').get_weights())
		predictor_model.get_layer('dragonn_batchnorm_3').trainable = False

		predictor_model.get_layer('dragonn_dense_1').set_weights(saved_model.get_layer('dragonn_dense_1_copy').get_weights())
		predictor_model.get_layer('dragonn_dense_1').trainable = False

	def _initialize_predictor_weights_old(predictor_model, saved_model=saved_model, model_path=model_path) :
		#Load pre-trained model
		#print(saved_model.summary())
		predictor_model.load_weights(model_path, by_name=True)

	def _load_predictor_func(sequence_input) :
		#DragoNN parameters
		seq_length = 145
		seq_input_shape = (seq_length, 4, 1)
		n_tasks = 1

		#Define model layers
		permute_input = Lambda(lambda x: x[..., -1])
		
		conv_1 = Conv1D(120, 5, padding='valid', activation='relu', name='dragonn_conv1d_1')
		batchnorm_1 = BatchNormalization(axis=-1, name='dragonn_batchnorm_1')
		drop_1 = Dropout(0.1)
		
		conv_2 = Conv1D(120, 5, padding='valid', activation='relu', name='dragonn_conv1d_2')
		batchnorm_2 = BatchNormalization(axis=-1, name='dragonn_batchnorm_2')
		drop_2 = Dropout(0.1)
		
		conv_3 = Conv1D(120, 5, padding='valid', activation='relu', name='dragonn_conv1d_3')
		batchnorm_3 = BatchNormalization(axis=-1, name='dragonn_batchnorm_3')
		drop_3 = Dropout(0.1)
		
		flatten_3 = Flatten()
		final_dense = Dense(12, name='dragonn_dense_1')

		#Execute functional model definition
		permuted_input = permute_input(sequence_input)
		
		drop_1_out = drop_1(batchnorm_1(conv_1(permuted_input), training=False), training=False)
		drop_2_out = drop_2(batchnorm_2(conv_2(drop_1_out), training=False), training=False)
		drop_3_out = drop_3(batchnorm_3(conv_3(drop_2_out), training=False), training=False)

		final_dense_out = final_dense(flatten_3(drop_3_out))

		predictor_inputs = []
		predictor_outputs = [final_dense_out, drop_1_out, drop_2_out, drop_3_out]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
