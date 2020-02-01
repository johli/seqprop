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
		predictor_model.get_layer('deepsea_conv2d_1').set_weights(saved_model.get_layer('11').get_weights())
		predictor_model.get_layer('deepsea_conv2d_1').trainable = False
		
		predictor_model.get_layer('deepsea_conv2d_2').set_weights(saved_model.get_layer('14').get_weights())
		predictor_model.get_layer('deepsea_conv2d_2').trainable = False
		
		predictor_model.get_layer('deepsea_conv2d_3').set_weights(saved_model.get_layer('17').get_weights())
		predictor_model.get_layer('deepsea_conv2d_3').trainable = False

		predictor_model.get_layer('deepsea_dense_1').set_weights(saved_model.get_layer('27').get_weights())
		predictor_model.get_layer('deepsea_dense_1').trainable = False

		predictor_model.get_layer('deepsea_dense_2').set_weights(saved_model.get_layer('29').get_weights())
		predictor_model.get_layer('deepsea_dense_2').trainable = False

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
		permute_input = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)))
		
		def permute_nt_func(pwm) :
			a_band = K.expand_dims(pwm[..., :, 0, :], axis=-2)
			c_band = K.expand_dims(pwm[..., :, 1, :], axis=-2)
			g_band = K.expand_dims(pwm[..., :, 2, :], axis=-2)
			t_band = K.expand_dims(pwm[..., :, 3, :], axis=-2)

			shuffled_pwm = K.concatenate([
				a_band,
				g_band,
				c_band,
				t_band
			], axis=-2)
			
			return shuffled_pwm
		
		permute_nt = Lambda(permute_nt_func)
		
		conv_1 = Conv2D(320, (1, 8), data_format='channels_first', padding='valid', activation='linear', name='deepsea_conv2d_1')
		relu_1 = Activation('relu')
		max_pool_1 = MaxPooling2D(pool_size=(1,4), data_format='channels_first')
		drop_1 = Dropout(0.2)
		
		conv_2 = Conv2D(480, (1, 8), data_format='channels_first', padding='valid', activation='linear', name='deepsea_conv2d_2')
		relu_2 = Activation('relu')
		max_pool_2 = MaxPooling2D(pool_size=(1,4), data_format='channels_first')
		drop_2 = Dropout(0.2)
		
		conv_3 = Conv2D(960, (1, 8), data_format='channels_first', padding='valid', activation='linear', name='deepsea_conv2d_3')
		relu_3 = Activation('relu')
		drop_3 = Dropout(0.5)
		
		flatten_3 = Flatten()
		dense_4 = Dense(925, name='deepsea_dense_1')
		relu_4 = Activation('relu')
		
		final_dense = Dense(919, name='deepsea_dense_2')
		final_sigm = Activation("sigmoid")
		final_clip = Lambda(lambda x: K.clip(x, K.epsilon(), 1. - K.epsilon()))

		#Execute functional model definition
		permuted_input = permute_input(permute_nt(sequence_input))
		
		drop_1_out = drop_1(max_pool_1(relu_1(conv_1(permuted_input))), training=False)
		drop_2_out = drop_2(max_pool_2(relu_2(conv_2(drop_1_out))), training=False)
		drop_3_out = drop_3(relu_3(conv_3(drop_2_out)))

		relu_4_out = relu_4(dense_4(flatten_3(drop_3_out)))
		
		final_dense_out = final_dense(relu_4_out)
		final_sigm_out = final_clip(final_sigm(final_dense_out))

		predictor_inputs = []
		predictor_outputs = [final_sigm_out, final_dense_out]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
