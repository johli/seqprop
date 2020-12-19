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

	# sublayer 1
	saved_model.add(Conv1D(48, 3, padding='same', activation='relu', input_shape=(145, 4), name='dragonn_conv1d_1_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_1_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_1_copy'))

	saved_model.add(Conv1D(64, 3, padding='same', activation='relu', name='dragonn_conv1d_2_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_2_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_2_copy'))

	saved_model.add(Conv1D(100, 3, padding='same', activation='relu', name='dragonn_conv1d_3_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_3_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_3_copy'))

	saved_model.add(Conv1D(150, 7, padding='same', activation='relu', name='dragonn_conv1d_4_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_4_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_4_copy'))

	saved_model.add(Conv1D(300, 7, padding='same', activation='relu', name='dragonn_conv1d_5_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_5_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_5_copy'))

	saved_model.add(MaxPooling1D(3))

	# sublayer 2
	saved_model.add(Conv1D(200, 7, padding='same', activation='relu', name='dragonn_conv1d_6_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_6_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_6_copy'))

	saved_model.add(Conv1D(200, 3, padding='same', activation='relu', name='dragonn_conv1d_7_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_7_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_7_copy'))

	saved_model.add(Conv1D(200, 3, padding='same', activation='relu', name='dragonn_conv1d_8_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_8_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_8_copy'))

	saved_model.add(MaxPooling1D(4))

	# sublayer 3
	saved_model.add(Conv1D(200, 7, padding='same', activation='relu', name='dragonn_conv1d_9_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_9_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_9_copy'))

	saved_model.add(MaxPooling1D(4))

	saved_model.add(Flatten())
	saved_model.add(Dense(100, activation='relu', name='dragonn_dense_1_copy'))
	saved_model.add(BatchNormalization(name='dragonn_batchnorm_10_copy'))
	saved_model.add(Dropout(0.1, name='dragonn_dropout_10_copy'))
	saved_model.add(Dense(12, activation='linear', name='dragonn_dense_2_copy'))

	saved_model.compile(
		loss= "mean_squared_error",
		optimizer=keras.optimizers.SGD(lr=0.1)
	)

	saved_model.load_weights(model_path)
	#print(saved_model.summary())

	def _initialize_predictor_weights(predictor_model, saved_model=saved_model) :
		#Load pre-trained model
		
		for i in range(1, 10) :
			predictor_model.get_layer('dragonn_conv1d_' + str(i)).set_weights(saved_model.get_layer('dragonn_conv1d_' + str(i) + '_copy').get_weights())
			predictor_model.get_layer('dragonn_conv1d_' + str(i)).trainable = False
			predictor_model.get_layer('dragonn_batchnorm_' + str(i)).set_weights(saved_model.get_layer('dragonn_batchnorm_' + str(i) + '_copy').get_weights())
			predictor_model.get_layer('dragonn_batchnorm_' + str(i)).trainable = False

		predictor_model.get_layer('dragonn_batchnorm_10').set_weights(saved_model.get_layer('dragonn_batchnorm_10_copy').get_weights())
		predictor_model.get_layer('dragonn_batchnorm_10').trainable = False

		predictor_model.get_layer('dragonn_dense_1').set_weights(saved_model.get_layer('dragonn_dense_1_copy').get_weights())
		predictor_model.get_layer('dragonn_dense_1').trainable = False
		
		predictor_model.get_layer('dragonn_dense_2').set_weights(saved_model.get_layer('dragonn_dense_2_copy').get_weights())
		predictor_model.get_layer('dragonn_dense_2').trainable = False

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
		
		conv_1 = Conv1D(48, 3, padding='same', activation='relu', name='dragonn_conv1d_1')
		batchnorm_1 = BatchNormalization(axis=-1, name='dragonn_batchnorm_1')
		drop_1 = Dropout(0.1)
		
		conv_2 = Conv1D(64, 3, padding='same', activation='relu', name='dragonn_conv1d_2')
		batchnorm_2 = BatchNormalization(axis=-1, name='dragonn_batchnorm_2')
		drop_2 = Dropout(0.1)
		
		conv_3 = Conv1D(100, 3, padding='same', activation='relu', name='dragonn_conv1d_3')
		batchnorm_3 = BatchNormalization(axis=-1, name='dragonn_batchnorm_3')
		drop_3 = Dropout(0.1)
		
		conv_4 = Conv1D(150, 7, padding='same', activation='relu', name='dragonn_conv1d_4')
		batchnorm_4 = BatchNormalization(axis=-1, name='dragonn_batchnorm_4')
		drop_4 = Dropout(0.1)
		
		conv_5 = Conv1D(300, 7, padding='same', activation='relu', name='dragonn_conv1d_5')
		batchnorm_5 = BatchNormalization(axis=-1, name='dragonn_batchnorm_5')
		drop_5 = Dropout(0.1)
		maxpool_5 = MaxPooling1D(3)
		
		conv_6 = Conv1D(200, 7, padding='same', activation='relu', name='dragonn_conv1d_6')
		batchnorm_6 = BatchNormalization(axis=-1, name='dragonn_batchnorm_6')
		drop_6 = Dropout(0.1)
		
		conv_7 = Conv1D(200, 3, padding='same', activation='relu', name='dragonn_conv1d_7')
		batchnorm_7 = BatchNormalization(axis=-1, name='dragonn_batchnorm_7')
		drop_7 = Dropout(0.1)
		
		conv_8 = Conv1D(200, 3, padding='same', activation='relu', name='dragonn_conv1d_8')
		batchnorm_8 = BatchNormalization(axis=-1, name='dragonn_batchnorm_8')
		drop_8 = Dropout(0.1)
		maxpool_8 = MaxPooling1D(4)
		
		conv_9 = Conv1D(200, 7, padding='same', activation='relu', name='dragonn_conv1d_9')
		batchnorm_9 = BatchNormalization(axis=-1, name='dragonn_batchnorm_9')
		drop_9 = Dropout(0.1)
		maxpool_9 = MaxPooling1D(4)
		
		flatten_9 = Flatten()
		dense_10 = Dense(100, activation='relu', name='dragonn_dense_1')
		batchnorm_10 = BatchNormalization(axis=-1, name='dragonn_batchnorm_10')
		drop_10 = Dropout(0.1)
		final_dense = Dense(12, name='dragonn_dense_2')

		#Execute functional model definition
		permuted_input = permute_input(sequence_input)
		
		drop_1_out = drop_1(batchnorm_1(conv_1(permuted_input), training=False), training=False)
		drop_2_out = drop_2(batchnorm_2(conv_2(drop_1_out), training=False), training=False)
		drop_3_out = drop_3(batchnorm_3(conv_3(drop_2_out), training=False), training=False)
		drop_4_out = drop_4(batchnorm_4(conv_4(drop_3_out), training=False), training=False)
		maxpool_5_out = maxpool_5(drop_5(batchnorm_5(conv_5(drop_4_out), training=False), training=False))
		
		drop_6_out = drop_6(batchnorm_6(conv_6(maxpool_5_out), training=False), training=False)
		drop_7_out = drop_7(batchnorm_7(conv_7(drop_6_out), training=False), training=False)
		maxpool_8_out = maxpool_8(drop_8(batchnorm_8(conv_8(drop_7_out), training=False), training=False))
		maxpool_9_out = maxpool_9(drop_9(batchnorm_9(conv_9(maxpool_8_out), training=False), training=False))
		
		drop_10_out = drop_10(batchnorm_10(dense_10(flatten_9(maxpool_9_out)), training=False), training=False)

		final_dense_out = final_dense(drop_10_out)

		predictor_inputs = []
		predictor_outputs = [final_dense_out, maxpool_5_out, maxpool_8_out, maxpool_9_out]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
