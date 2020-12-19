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


#Saved Model definition

def load_saved_predictor(model_path, library_context=None) :
	
	def dummy_pred_func(y_true, y_pred) :
		return y_pred

	n_models = 1
	saved_oracles = []
	for model_ix in range(n_models) :
		model_suffix = "_n_models_" + str(n_models) + "_model_ix_" + str(model_ix)
		
		saved_oracle = load_model(model_path + model_suffix + ".h5", custom_objects={'neg_log_likelihood' : lambda y_true, y_pred: y_pred})
		saved_oracles.append(saved_oracle)
	
	#print(saved_model.summary())

	def _initialize_predictor_weights(predictor_model, saved_oracles=saved_oracles) :
		#Load pre-trained weights
		for i in range(len(saved_oracles)) :
			#print(saved_oracles[i].summary())
			
			predictor_model.get_layer('weak_conv2d_1_' + str(i) + '_' + str(len(saved_oracles))).set_weights(saved_oracles[i].get_layer('weak_conv2d_1').get_weights())
			predictor_model.get_layer('weak_conv2d_1_' + str(i) + '_' + str(len(saved_oracles))).trainable = False
			
			predictor_model.get_layer('weak_batchnorm_1_' + str(i) + '_' + str(len(saved_oracles))).set_weights(saved_oracles[i].get_layer('weak_batchnorm_1').get_weights())
			predictor_model.get_layer('weak_batchnorm_1_' + str(i) + '_' + str(len(saved_oracles))).trainable = False
			
			predictor_model.get_layer('weak_conv2d_2_' + str(i) + '_' + str(len(saved_oracles))).set_weights(saved_oracles[i].get_layer('weak_conv2d_2').get_weights())
			predictor_model.get_layer('weak_conv2d_2_' + str(i) + '_' + str(len(saved_oracles))).trainable = False
			
			predictor_model.get_layer('weak_batchnorm_2_' + str(i) + '_' + str(len(saved_oracles))).set_weights(saved_oracles[i].get_layer('weak_batchnorm_2').get_weights())
			predictor_model.get_layer('weak_batchnorm_2_' + str(i) + '_' + str(len(saved_oracles))).trainable = False
			
			predictor_model.get_layer('weak_dense_1_' + str(i) + '_' + str(len(saved_oracles))).set_weights(saved_oracles[i].get_layer('weak_dense_1').get_weights())
			predictor_model.get_layer('weak_dense_1_' + str(i) + '_' + str(len(saved_oracles))).trainable = False
			
			predictor_model.get_layer('weak_dense_2_' + str(i) + '_' + str(len(saved_oracles))).set_weights(saved_oracles[i].get_layer('weak_dense_2').get_weights())
			predictor_model.get_layer('weak_dense_2_' + str(i) + '_' + str(len(saved_oracles))).trainable = False

	def _load_predictor_func(sequence_input, n_models=n_models) :
		#Network parameters
		seq_length = 157
		seq_input_shape = (seq_length, 4, 1)
		
		#Build single model
		def build_model(x, i, num_models) :
			
			conv_1 = Conv2D(32, (1, 8), padding='valid', activation='linear', name='weak_conv2d_1_' + str(i) + '_' + str(num_models))
			batchnorm_1 = BatchNormalization(axis=-1, name='weak_batchnorm_1_' + str(i) + '_' + str(num_models))
			relu_1 = Activation('relu')

			conv_2 = Conv2D(64, (1, 7), padding='valid', activation='linear', name='weak_conv2d_2_' + str(i) + '_' + str(num_models))
			batchnorm_2 = BatchNormalization(axis=-1, name='weak_batchnorm_2_' + str(i) + '_' + str(num_models))
			relu_2 = Activation('relu')

			max_pool_3 = MaxPooling2D(pool_size=(1, 4))

			flatten_3 = Flatten()
			dense_4 = Dense(64, activation='elu', name='weak_dense_1_' + str(i) + '_' + str(num_models))

			final_dense = Dense(2, activation='linear', name='weak_dense_2_' + str(i) + '_' + str(num_models))
			final_act = Activation("linear")

			relu_1_out = relu_1(batchnorm_1(conv_1(Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)))(x)), training=False))
			relu_2_out = relu_2(batchnorm_2(conv_2(relu_1_out), training=False))

			max_pool_3_out = max_pool_3(relu_2_out)

			dense_4_out = dense_4(flatten_3(max_pool_3_out))

			final_dense_out = final_dense(dense_4_out)
			y = final_act(final_dense_out)
			
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
			oracles_means = Lambda(lambda y: K.expand_dims(y[:, 0, 0], axis=-1))(oracles[0])
			oracles_vars = Lambda(lambda y: K.expand_dims(y[:, 1, 0], axis=-1))(oracles[0])

		predictor_inputs = []
		predictor_outputs = [oracles_mean, oracles_var, oracles_means, oracles_vars]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
