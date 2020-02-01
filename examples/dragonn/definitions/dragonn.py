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

	saved_model = load_model(model_path, custom_objects={
		'ambig_binary_crossentropy' : dummy_pred_func,
		'precision' : dummy_pred_func,
		'recall' : dummy_pred_func,
		'specificity' : dummy_pred_func,
		'fpr' : dummy_pred_func,
		'fnr' : dummy_pred_func,
		'fdr' : dummy_pred_func,
		'f1' : dummy_pred_func
	})
	#print(saved_model.summary())

	def _initialize_predictor_weights(predictor_model, saved_model=saved_model) :
		#Load pre-trained model
		predictor_model.get_layer('dragonn_conv2d_1').set_weights(saved_model.get_layer('conv2d_1').get_weights())
		predictor_model.get_layer('dragonn_conv2d_1').trainable = False
		predictor_model.get_layer('dragonn_batchnorm_1').set_weights(saved_model.get_layer('batch_normalization_1').get_weights())
		predictor_model.get_layer('dragonn_batchnorm_1').trainable = False

		predictor_model.get_layer('dragonn_conv2d_2').set_weights(saved_model.get_layer('conv2d_2').get_weights())
		predictor_model.get_layer('dragonn_conv2d_2').trainable = False
		predictor_model.get_layer('dragonn_batchnorm_2').set_weights(saved_model.get_layer('batch_normalization_2').get_weights())
		predictor_model.get_layer('dragonn_batchnorm_2').trainable = False
		
		predictor_model.get_layer('dragonn_conv2d_3').set_weights(saved_model.get_layer('conv2d_3').get_weights())
		predictor_model.get_layer('dragonn_conv2d_3').trainable = False
		predictor_model.get_layer('dragonn_batchnorm_3').set_weights(saved_model.get_layer('batch_normalization_3').get_weights())
		predictor_model.get_layer('dragonn_batchnorm_3').trainable = False

		predictor_model.get_layer('dragonn_dense_1').set_weights(saved_model.get_layer('dense_1').get_weights())
		predictor_model.get_layer('dragonn_dense_1').trainable = False
		predictor_model.get_layer('dragonn_batchnorm_4').set_weights(saved_model.get_layer('batch_normalization_4').get_weights())
		predictor_model.get_layer('dragonn_batchnorm_4').trainable = False

		predictor_model.get_layer('dragonn_dense_2').set_weights(saved_model.get_layer('dense_2').get_weights())
		predictor_model.get_layer('dragonn_dense_2').trainable = False

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
		permute_input = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)))
		
		conv_1 = Conv2D(50, (1, 15), padding='same', activation='linear', name='dragonn_conv2d_1')
		batchnorm_1 = BatchNormalization(axis=-1, name='dragonn_batchnorm_1')
		relu_1 = Activation('relu')
		
		conv_2 = Conv2D(50, (1, 15), padding='same', activation='linear', name='dragonn_conv2d_2')
		batchnorm_2 = BatchNormalization(axis=-1, name='dragonn_batchnorm_2')
		relu_2 = Activation('relu')
		
		conv_3 = Conv2D(50, (1, 13), padding='same', activation='linear', name='dragonn_conv2d_3')
		batchnorm_3 = BatchNormalization(axis=-1, name='dragonn_batchnorm_3')
		relu_3 = Activation('relu')
		
		max_pool_3 = MaxPooling2D(pool_size=(1,40))
		
		flatten_3 = Flatten()
		dense_4 = Dense(50, name='dragonn_dense_1')
		batchnorm_4 = BatchNormalization(axis=-1, name='dragonn_batchnorm_4')
		relu_4 = Activation('relu')
		drop_4 = Dropout(0.2)
		
		final_dense = Dense(n_tasks, name='dragonn_dense_2')
		final_sigm = Activation("sigmoid")
		final_clip = Lambda(lambda x: K.clip(x, K.epsilon(), 1. - K.epsilon()))

		#Execute functional model definition
		permuted_input = permute_input(sequence_input)
		
		relu_1_out = relu_1(batchnorm_1(conv_1(permuted_input), training=False))
		relu_2_out = relu_2(batchnorm_2(conv_2(relu_1_out), training=False))
		relu_3_out = relu_3(batchnorm_3(conv_3(relu_2_out), training=False))

		max_pool_3_out = max_pool_3(relu_3_out)
		
		drop_4_out = drop_4(relu_4(batchnorm_4(dense_4(flatten_3(max_pool_3_out)), training=False)), training=False)
		
		final_dense_out = final_dense(drop_4_out)
		final_sigm_out = final_clip(final_sigm(final_dense_out))

		predictor_inputs = []
		predictor_outputs = [final_sigm_out, final_dense_out]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
