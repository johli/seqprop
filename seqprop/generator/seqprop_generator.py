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
from tensorflow.python.framework import ops

import isolearn.keras as iso

import numpy as np

#Stochastic Binarized Neuron helper functions (Tensorflow)
#ST Estimator code adopted from https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html
#See Github https://github.com/spitis/

def st_sampled_softmax(logits):
	with ops.name_scope("STSampledSoftmax") as namescope :
		nt_probs = tf.nn.softmax(logits)
		onehot_dim = logits.get_shape().as_list()[1]
		sampled_onehot = tf.one_hot(tf.squeeze(tf.multinomial(logits, 1), 1), onehot_dim, 1.0, 0.0)
		with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
			return tf.ceil(sampled_onehot * nt_probs)

def st_hardmax_softmax(logits):
	with ops.name_scope("STHardmaxSoftmax") as namescope :
		nt_probs = tf.nn.softmax(logits)
		onehot_dim = logits.get_shape().as_list()[1]
		sampled_onehot = tf.one_hot(tf.argmax(nt_probs, 1), onehot_dim, 1.0, 0.0)
		with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
			return tf.ceil(sampled_onehot * nt_probs)

@ops.RegisterGradient("STMul")
def st_mul(op, grad):
	return [grad, grad]

#PWM Masking and Sampling helper functions

def mask_pwm(inputs) :
	pwm, onehot_template, onehot_mask = inputs

	return pwm * onehot_mask + onehot_template

def sample_pwm(pwm_logits) :
	n_sequences = pwm_logits.get_shape().as_list()[0]
	seq_length = pwm_logits.get_shape().as_list()[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 4))
	sampled_pwm = K.switch(K.learning_phase(), st_sampled_softmax(flat_pwm), st_hardmax_softmax(flat_pwm))

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 4, 1))

def max_pwm(pwm_logits) :
	n_sequences = pwm_logits.get_shape().as_list()[0]
	seq_length = pwm_logits.get_shape().as_list()[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 4))
	sampled_pwm = st_hardmax_softmax(flat_pwm)

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 4, 1))


#SeqProp helper functions

def initialize_sequence_templates(generator, sequence_templates) :

	encoder = iso.OneHotEncoder(seq_length=len(sequence_templates[0]))
	onehot_templates = np.concatenate([encoder(sequence_template).reshape((1, len(sequence_template), 4, 1)) for sequence_template in sequence_templates], axis=0)

	for i in range(len(sequence_templates)) :
		sequence_template = sequence_templates[i]

		for j in range(len(sequence_template)) :
			if sequence_template[j] != 'N' :
				if sequence_template[j] != 'X' :
					nt_ix = np.argmax(onehot_templates[i, j, :, 0])
					onehot_templates[i, j, :, :] = -4
					onehot_templates[i, j, nt_ix, :] = 10
				else :
					onehot_templates[i, j, :, :] = -1

	onehot_masks = np.zeros((len(sequence_templates), len(sequence_templates[0]), 4, 1))
	for i in range(len(sequence_templates)) :
		sequence_template = sequence_templates[i]

		for j in range(len(sequence_template)) :
			if sequence_template[j] == 'N' :
				onehot_masks[i, j, :, :] = 1.0


	generator.get_layer('template_dense').set_weights([onehot_templates.reshape(1, -1)])
	generator.get_layer('template_dense').trainable = False

	generator.get_layer('mask_dense').set_weights([onehot_masks.reshape(1, -1)])
	generator.get_layer('mask_dense').trainable = False

def initialize_sequences(generator, init_sequences, p_init) :

	encoder = iso.OneHotEncoder(seq_length=len(init_sequences[0]))
	onehot_sequences = np.concatenate([encoder(init_sequence).reshape((1, len(init_sequence), 4, 1)) for init_sequence in init_sequences], axis=0)

	onehot_logits = generator.get_layer('policy_pwm').reshape((len(init_sequences), len(init_sequences[0]), 4, 1))

	on_logit = np.log(p_init / (1. - p_init))

	p_off = (1. - p_init) / 3.
	off_logit = np.log(p_off / (1. - p_off))

	for i in range(len(init_sequences)) :
		init_sequence = init_sequences[i]

		for j in range(len(init_sequence)) :
			if init_sequence[j] == 'A' :
				nt_ix = 0
			elif init_sequence[j] == 'C' :
				nt_ix = 1
			elif init_sequence[j] == 'G' :
				nt_ix = 2
			elif init_sequence[j] == 'T' :
				nt_ix = 3

			onehot_logits[i, j, :, :] = off_logit
			onehot_logits[i, j, nt_ix, :] = on_logit

	generator.get_layer('policy_pwm').set_weights([onehot_logits.reshape(1, -1)])



#SeqProp Generator Model definitions

#Generator that samples a single one-hot sequence per trainable PWM
def build_generator(seq_length, n_sequences=1, n_samples=None, sequence_templates=None, init_sequences=None, p_init=0.5, batch_normalize_pwm=False) :

	use_samples = True
	if n_samples is None :
		use_samples = False
		n_samples = 1

	#Seed input for all dense/embedding layers
	ones_input = Input(tensor=K.ones((1, 1)), name='seed_input')

	#Initialize a Lambda layer to reshape flat matrices into PWM tensors
	reshape_layer = Lambda(lambda x: K.reshape(x, (n_sequences, seq_length, 4, 1)), name='onehot_reshape')
	
	#Initialize Template, Masking and Trainable PWMs
	onehot_template_dense = Dense(n_sequences * seq_length * 4, use_bias=False, kernel_initializer='zeros', name='template_dense')
	onehot_mask_dense = Dense(n_sequences * seq_length * 4, use_bias=False, kernel_initializer='ones', name='mask_dense')
	dense_seq_layer = Dense(n_sequences * seq_length * 4, use_bias=False, kernel_initializer='glorot_uniform', name='policy_pwm')
	
	#Initialize Templating and Masking Lambda layer
	masking_layer = Lambda(mask_pwm, output_shape = (seq_length, 4, 1), name='masking_layer')
	
	#Get Template, Mask and Trainable PWM logits
	onehot_template = reshape_layer(onehot_template_dense(ones_input))
	onehot_mask = reshape_layer(onehot_mask_dense(ones_input))
	onehot_logits = reshape_layer(dense_seq_layer(ones_input))

	#Batch Normalize PWM Logits
	if batch_normalize_pwm :
	   onehot_logits = BatchNormalization(axis=2, name='policy_batch_norm')(onehot_logits)
	
	#Add Template and Multiply Mask
	pwm_logits = masking_layer([onehot_logits, onehot_template, onehot_mask])
	
	#Get PWM from logits
	pwm = Softmax(axis=-2, name='pwm')(pwm_logits)

	#Optionally tile each PWM to sample from
	if use_samples :
		pwm_logits = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(pwm_logits)
	
	#Sample proper One-hot coded sequences from PWMs
	sampled_pwm = Lambda(sample_pwm, name='pwm_sampler')(pwm_logits)
	
	#Optionally create sample axis
	if use_samples :
		sampled_pwm = Lambda(lambda x: K.reshape(x, (n_samples, n_sequences, seq_length, 4, 1)))(sampled_pwm)

	generator_model = Model(
		inputs=[
			ones_input		#Dummy Seed Input
		],
		outputs=[
			pwm_logits,		#Logits of the Templated and Masked PWMs
			pwm,			#Templated and Masked PWMs
			sampled_pwm		#Sampled One-hot sequences (n_samples per trainable PWM)
		]
	)

	if sequence_templates is not None :
		initialize_sequence_templates(generator_model, sequence_templates)

	if init_sequences is not None :
		initialize_sequences(generator, init_sequences, p_init)

	#Lock all generator layers except policy layers
	for generator_layer in generator_model.layers :
		generator_layer.trainable = False
		
		if 'policy' in generator_layer.name :
			generator_layer.trainable = True

	return 'seqprop_generator', generator_model

#(Re-)Initialize PWM weights
def reset_generator(generator_model) :
    session = K.get_session()
    for generator_layer in generator_model.layers :
        if 'policy' in generator_layer.name :
            for v in generator_layer.__dict__:
                v_arg = getattr(generator_layer, v)
                if hasattr(v_arg,'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    print('reinitializing layer {}.{}'.format(generator_layer.name, v))
