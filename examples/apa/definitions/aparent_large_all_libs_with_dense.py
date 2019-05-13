import keras
from keras.models import Sequential, Model, load_model
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


#APARENT Saved Model definition

def load_saved_predictor(model_path, library_context=None) :

	saved_model = load_model(model_path)

	hardcoded_lib = np.zeros((1, 13))
	if library_context is not None :
		#Library index
		libraries = ['tomm5_up_n20c20_dn_c20', 'tomm5_up_c20n20_dn_c20', 'tomm5_up_n20c20_dn_n20', 'tomm5_up_c20n20_dn_n20', 'doubledope', 'simple', 'atr', 'hsp', 'snh', 'sox', 'wha', 'array', 'aar']
		library_dict = {lib : i for i, lib in enumerate(libraries)}

		if library_context in library_dict :
			lib_ix = library_dict[library_context]
			hardcoded_lib[0, lib_ix] = 1.


	def _initialize_predictor_weights(predictor_model, saved_model=saved_model) :
		#Load pre-trained model
		predictor_model.get_layer('aparent_conv_1').set_weights(saved_model.get_layer('conv2d_1').get_weights())
		predictor_model.get_layer('aparent_conv_1').trainable = False

		predictor_model.get_layer('aparent_conv_2').set_weights(saved_model.get_layer('conv2d_2').get_weights())
		predictor_model.get_layer('aparent_conv_2').trainable = False

		predictor_model.get_layer('aparent_dense_1').set_weights(saved_model.get_layer('dense_1').get_weights())
		predictor_model.get_layer('aparent_dense_1').trainable = False

		predictor_model.get_layer('aparent_dense_2').set_weights(saved_model.get_layer('dense_2').get_weights())
		predictor_model.get_layer('aparent_dense_2').trainable = False

		predictor_model.get_layer('aparent_cut_dense').set_weights(saved_model.get_layer('dense_3').get_weights())
		predictor_model.get_layer('aparent_cut_dense').trainable = False

		predictor_model.get_layer('aparent_iso_dense').set_weights(saved_model.get_layer('dense_4').get_weights())
		predictor_model.get_layer('aparent_iso_dense').trainable = False

	def _load_predictor_func(sequence_input) :
		#APARENT parameters
		seq_length = 205
		seq_input_shape = (205, 4, 1)
		#lib_input_shape = (n_sequences * n_samples, 13)
		#distal_pas_shape = (n_sequences * n_samples, 1)
		num_outputs_iso = 1
		num_outputs_cut = 206

		
		#Shared model definition
		layer_1 = Conv2D(96, (8, 4), padding='valid', activation='relu', name='aparent_conv_1')
		layer_1_pool = MaxPooling2D(pool_size=(2, 1))
		layer_2 = Conv2D(128, (6, 1), padding='valid', activation='relu', name='aparent_conv_2')
		layer_dense = Dense(512, activation='relu', name='aparent_dense_1')
		layer_drop = Dropout(0.2)
		layer_dense2 = Dense(256, activation='relu', name='aparent_dense_2')
		layer_drop2 = Dropout(0.2)

		def shared_model(seq_input, distal_pas_input) :
			return layer_drop2(
						layer_dense2(
							layer_drop(
								layer_dense(
									Concatenate()([
										Flatten()(
											layer_2(
												layer_1_pool(
													layer_1(
														seq_input
													)
												)
											)
										),
										distal_pas_input
									])
								), training=False
							)
						), training=False
					)

		
		#lib_input = Input(tensor=K.zeros(lib_input_shape))
		lib_input = Lambda(lambda x: K.tile(K.variable(hardcoded_lib), (K.shape(x)[0], 1)))(sequence_input)

		#distal_pas_input = Input(tensor=K.ones(distal_pas_shape))
		distal_pas_input = Lambda(lambda x: K.tile(K.variable(np.ones((1, 1))), (K.shape(x)[0], 1)))(sequence_input)

		#Outputs
		plasmid_out_shared = Concatenate()([shared_model(sequence_input, distal_pas_input), lib_input])

		plasmid_score_cut = Dense(num_outputs_cut, kernel_initializer='zeros', name='aparent_cut_dense')(plasmid_out_shared)
		plasmid_score_iso = Dense(num_outputs_iso, kernel_initializer='zeros', name='aparent_iso_dense')(plasmid_out_shared)
		
		plasmid_out_cut = Softmax(axis=-1)(plasmid_score_cut)
		plasmid_out_iso = Dense(num_outputs_iso, activation='sigmoid', kernel_initializer='ones', use_bias=False)(plasmid_score_iso)
		

		#Initialize pre-trained predictor weights
		#_initialize_predictor_weights(predictor_model, saved_model)

		
		predictor_inputs = []#[lib_input, distal_pas_input]
		predictor_outputs = [plasmid_out_iso, plasmid_out_cut, plasmid_score_iso, plasmid_score_cut, plasmid_out_shared]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
