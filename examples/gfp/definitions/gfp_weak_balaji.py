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

def build_model(M):
	x = Input(shape=(M, 20,))
	y = Flatten()(x)
	y = Dense(50, activation='elu')(y)
	y = Dense(2)(y)
	model = Model(inputs=x, outputs=y)
	return model

#Saved Model definition

def load_saved_predictor(model_path, oracle_suffix, random_state, num_models) :
	
	saved_oracles = [build_model(237) for i in range(num_models)]
	for i in range(num_models) :
		saved_oracles[i].load_weights(model_path + "oracle_%i%s.h5" % (random_state, oracle_suffix))

	def _initialize_predictor_weights(predictor_model, saved_oracles=saved_oracles) :
		#Load pre-trained weights
		for i in range(len(saved_oracles)) :
			#print(saved_oracles[i].summary())
			
			dense_1_name = 'dense_1'
			dense_2_name = 'dense_2'
			
			curr_dense_found = 0
			for saved_layer in saved_oracles[i].layers :
				if 'dense_' in saved_layer.name :
					if curr_dense_found == 0 :
						dense_1_name = saved_layer.name
						curr_dense_found += 1
					elif curr_dense_found == 1 :
						dense_2_name = saved_layer.name
						curr_dense_found += 1
			
			predictor_model.get_layer('gfp_' + str(i) + '_' + str(num_models) + '_dense_1').set_weights(saved_oracles[i].get_layer(dense_1_name).get_weights())
			predictor_model.get_layer('gfp_' + str(i) + '_' + str(num_models) + '_dense_1').trainable = False
			
			predictor_model.get_layer('gfp_' + str(i) + '_' + str(num_models) + '_dense_2').set_weights(saved_oracles[i].get_layer(dense_2_name).get_weights())
			predictor_model.get_layer('gfp_' + str(i) + '_' + str(num_models) + '_dense_2').trainable = False

	def _load_predictor_func(sequence_input, n_models=num_models) :
		#Network parameters
		seq_length = 2377
		seq_input_shape = (seq_length, 4, 1)
		
		#Build single model
		def build_model(x, i, num_models) :
			y = Flatten()(x)
			y = Dense(50, activation='elu', name='gfp_' + str(i) + '_' + str(num_models) + '_dense_1')(y)
			y = Dense(2, name='gfp_' + str(i) + '_' + str(num_models) + '_dense_2')(y)
			y = Lambda(lambda yy: K.concatenate([K.expand_dims(K.expand_dims(yy[:, 0], axis=-1), axis=-1), K.expand_dims(K.expand_dims(K.log(1.+K.exp(yy[:, 1])) + K.epsilon(), axis=-1), axis=-1)], axis=1))(y)
			return y
		
		oracles = [build_model(sequence_input, i, n_models) for i in range(n_models)]
		
		oracles_mean = None
		oracles_var = None
		oracles_means = None
		oracles_vars = None
		if len(oracles) > 1 :
			oracles_concat = Concatenate(axis=-1)(oracles)
			
			oracles_means = Lambda(lambda y: y[:, 0, :])(oracles_concat)
			oracles_vars = Lambda(lambda y: y[:, 1, :])(oracles_concat)
			
			oracles_mean = Lambda(lambda y: K.expand_dims(K.mean(y, axis=-1), axis=-1))(oracles_means)
			oracles_var = Lambda(lambda l: (1. / K.constant(n_models)) * (K.expand_dims(K.sum(l[1], axis=-1), axis=-1) + K.expand_dims(K.sum(l[0]**2, axis=-1), axis=-1)) - l[2]**2)([oracles_means, oracles_vars, oracles_mean])
		else :
			oracles_mean = Lambda(lambda y: K.expand_dims(y[:, 0, 0], axis=-1))(oracles[0])
			oracles_var = Lambda(lambda y: K.expand_dims(y[:, 1, 0], axis=-1))(oracles[0])

		predictor_inputs = []
		predictor_outputs = [oracles_mean, oracles_var, oracles_means, oracles_vars]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
