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

#SeqProp Predictor helper functions


#SeqProp Predictor Model definitions

#Predictor that predicts the function of the generated input sequence
def build_predictor(generator_model, load_predictor_function, n_sequences=1, n_samples=None, eval_mode='pwm') :

	use_samples = True
	if n_samples is None :
		use_samples = False
		n_samples = 1

	#Get PWM outputs from Generator Model
	pwm = generator_model.outputs[1]
	sampled_pwm = generator_model.outputs[2]

	seq_input = None
	if eval_mode == 'pwm' :
		seq_input = pwm
	elif eval_mode == 'sample' :
		seq_input = sampled_pwm
		if use_samples :
			seq_input = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] * K.shape(x)[1], K.shape(x)[2], K.shape(x)[3], K.shape(x)[4])))(seq_input)

	predictor_inputs, predictor_outputs, post_compile_function = load_predictor_function(seq_input)
	
	
	#Optionally create sample axis
	if use_samples :
		predictor_outputs = [
			Lambda(lambda x: K.reshape(x, (n_samples, n_sequences, K.shape(x)[-1])))(predictor_output)
			for predictor_output in predictor_outputs
		]

	predictor_model = Model(
		inputs = generator_model.inputs + predictor_inputs,
		outputs = generator_model.outputs + predictor_outputs
	)

	post_compile_function(predictor_model)

	#Lock all layers except policy layers
	for predictor_layer in predictor_model.layers :
		predictor_layer.trainable = False
		
		if 'policy' in predictor_layer.name :
			predictor_layer.trainable = True

	return 'seqprop_predictor_aparent_large', predictor_model
