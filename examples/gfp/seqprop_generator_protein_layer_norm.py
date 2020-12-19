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

from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints

class InstanceNormalization(Layer):
	def __init__(self,
				 axis=None,
				 epsilon=1e-3,
				 center=True,
				 scale=True,
				 beta_initializer='zeros',
				 gamma_initializer='ones',
				 beta_regularizer=None,
				 gamma_regularizer=None,
				 beta_constraint=None,
				 gamma_constraint=None,
				 **kwargs):
		super(InstanceNormalization, self).__init__(**kwargs)
		self.supports_masking = True
		self.axis = axis
		self.epsilon = epsilon
		self.center = center
		self.scale = scale
		self.beta_initializer = initializers.get(beta_initializer)
		self.gamma_initializer = initializers.get(gamma_initializer)
		self.beta_regularizer = regularizers.get(beta_regularizer)
		self.gamma_regularizer = regularizers.get(gamma_regularizer)
		self.beta_constraint = constraints.get(beta_constraint)
		self.gamma_constraint = constraints.get(gamma_constraint)

	def build(self, input_shape):
		ndim = len(input_shape)
		if self.axis == 0:
			raise ValueError('Axis cannot be zero')

		if (self.axis is not None) and (ndim == 2):
			raise ValueError('Cannot specify axis for rank 1 tensor')

		self.input_spec = InputSpec(ndim=ndim)

		if self.axis is None:
			shape = (1,)
		else:
			shape = (input_shape[self.axis],)

		if self.scale:
			self.gamma = self.add_weight(shape=shape,
										 name='gamma',
										 initializer=self.gamma_initializer,
										 regularizer=self.gamma_regularizer,
										 constraint=self.gamma_constraint)
		else:
			self.gamma = None
		if self.center:
			self.beta = self.add_weight(shape=shape,
										name='beta',
										initializer=self.beta_initializer,
										regularizer=self.beta_regularizer,
										constraint=self.beta_constraint)
		else:
			self.beta = None
		self.built = True

	def call(self, inputs, training=None):
		input_shape = K.int_shape(inputs)
		reduction_axes = list(range(0, len(input_shape)))

		if self.axis is not None:
			del reduction_axes[self.axis]

		del reduction_axes[0]

		mean = K.mean(inputs, reduction_axes, keepdims=True)
		stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
		normed = (inputs - mean) / stddev

		broadcast_shape = [1] * len(input_shape)
		if self.axis is not None:
			broadcast_shape[self.axis] = input_shape[self.axis]

		if self.scale:
			broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
			normed = normed * broadcast_gamma
		if self.center:
			broadcast_beta = K.reshape(self.beta, broadcast_shape)
			normed = normed + broadcast_beta
		return normed

	def get_config(self):
		config = {
			'axis': self.axis,
			'epsilon': self.epsilon,
			'center': self.center,
			'scale': self.scale,
			'beta_initializer': initializers.serialize(self.beta_initializer),
			'gamma_initializer': initializers.serialize(self.gamma_initializer),
			'beta_regularizer': regularizers.serialize(self.beta_regularizer),
			'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
			'beta_constraint': constraints.serialize(self.beta_constraint),
			'gamma_constraint': constraints.serialize(self.gamma_constraint)
		}
		base_config = super(InstanceNormalization, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

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

def st_sampled(logits):
	with ops.name_scope("STSampled") as namescope :
		#nt_probs = tf.nn.softmax(logits)
		onehot_dim = logits.get_shape().as_list()[1]
		sampled_onehot = tf.one_hot(tf.squeeze(tf.multinomial(logits, 1), 1), onehot_dim, 1.0, 0.0)
		with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul', 'Softmax' : 'Identity'}):
			return tf.ceil(sampled_onehot * tf.nn.softmax(logits))

#PWM Masking and Sampling helper functions

def mask_pwm(inputs) :
	pwm, onehot_template, onehot_mask = inputs

	return pwm * onehot_mask + onehot_template

def sample_pwm_only(pwm_logits) :
	n_sequences = pwm_logits.get_shape().as_list()[0]
	seq_length = pwm_logits.get_shape().as_list()[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 20))
	sampled_pwm = st_sampled_softmax(flat_pwm)

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 20, 1))

def sample_pwm_simple(pwm_logits) :
	n_sequences = pwm_logits.get_shape().as_list()[0]
	seq_length = pwm_logits.get_shape().as_list()[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 20))
	sampled_pwm = st_sampled(flat_pwm)

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 20, 1))

def sample_pwm(pwm_logits) :
	n_sequences = pwm_logits.get_shape().as_list()[0]
	seq_length = pwm_logits.get_shape().as_list()[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 20))
	sampled_pwm = K.switch(K.learning_phase(), st_sampled_softmax(flat_pwm), st_hardmax_softmax(flat_pwm))

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 20, 1))

def max_pwm(pwm_logits) :
	n_sequences = pwm_logits.get_shape().as_list()[0]
	seq_length = pwm_logits.get_shape().as_list()[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 20))
	sampled_pwm = st_hardmax_softmax(flat_pwm)

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 20, 1))

#Gumbel-Softmax (The Concrete Distribution) for annealed nucleotide sampling

def gumbel_softmax(logits, temperature=0.1) :
	gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
	batch_dim = logits.get_shape().as_list()[0]
	onehot_dim = logits.get_shape().as_list()[1]
	return gumbel_dist.sample()

def sample_gumbel(pwm_logits) :
	n_sequences = K.shape(pwm_logits)[0]
	seq_length = K.shape(pwm_logits)[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 20))
	sampled_pwm = gumbel_softmax(flat_pwm, temperature=0.1)

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 20, 1))

def sample_2(pwm_logits) :
	n_sequences = K.shape(pwm_logits)[0]
	seq_length = K.shape(pwm_logits)[1]
	
	U = tf.random.uniform(tf.shape(pwm_logits), minval=0, maxval=1)
	
	y_pssm = tf.nn.softmax(pwm_logits - tf.math.log(-tf.math.log(U + 1e-8) + 1e-8), axis=-2)
	#y_pssm = K.switch(train, y_pssm_sampled, tf.nn.softmax(pwm_logits,-2))
	
	y_seq = K.permute_dimensions(tf.one_hot(tf.argmax(y_pssm, axis=-2), 20, axis=-1), (0, 1, 3, 2))
	y_seq = tf.stop_gradient(y_seq - y_pssm) + y_pssm
	
	return y_seq

def max_2(pwm_logits) :
	n_sequences = K.shape(pwm_logits)[0]
	seq_length = K.shape(pwm_logits)[1]
	
	y_pssm = tf.nn.softmax(pwm_logits,-2)
	
	y_seq = tf.one_hot(tf.argmax(y_pssm, axis=-2), 20, axis=-2)
	y_seq = tf.stop_gradient(y_seq - y_pssm) + y_pssm
	
	return y_seq

#SeqProp helper functions

#SeqProp Generator Model definitions

#Generator that samples a single one-hot sequence per trainable PWM
def build_generator(seq_length, n_sequences=1, n_samples=None, batch_normalize_pwm=False, pwm_transform_func=None, validation_sample_mode='max', master_generator=None, logit_init_mode='glorot_uniform') :

	use_samples = True
	if n_samples is None :
		use_samples = False
		n_samples = 1

	#Seed input for all dense/embedding layers
	ones_input = Input(tensor=K.ones((1, 1)), name='seed_input')

	#Initialize a Lambda layer to reshape flat matrices into PWM tensors
	reshape_layer = Lambda(lambda x: K.reshape(x, (n_sequences, seq_length, 20, 1)), name='onehot_reshape')
	
	#Initialize Template, Masking and Trainable PWMs
	
	kernel_initializer = 'glorot_uniform'
	if logit_init_mode in ['glorot_uniform', 'glorot_normal'] :
		kernel_initializer = logit_init_mode
	elif 'random_normal_' in logit_init_mode :
		stddev = logit_init_mode[14:]
		kernel_initializer = initializers.RandomNormal(stddev=float(stddev))
	elif 'random_uniform_' in logit_init_mode :
		minval, maxval = logit_init_mode[15:].split("_")
		kernel_initializer = initializers.RandomUniform(minval=float(minval), maxval=float(maxval))
	
	dense_seq_layer = Dense(n_sequences * seq_length * 20, use_bias=False, kernel_initializer=kernel_initializer, name='policy_pwm')

	if master_generator is not None :
		dense_seq_layer = master_generator.get_layer('policy_pwm')
	
	#Get Template, Mask and Trainable PWM logits
	onehot_logits = reshape_layer(dense_seq_layer(ones_input))

	#Batch Normalize PWM Logits
	if batch_normalize_pwm :
		pwm_norm_layer = InstanceNormalization(axis=None, name='policy_batch_norm')
		if master_generator is not None :
			pwm_norm_layer = master_generator.get_layer('policy_batch_norm')
		onehot_logits = pwm_norm_layer(onehot_logits)
	
	#Add Template and Multiply Mask
	pwm_logits = onehot_logits
	
	#Get PWM from logits
	pwm = Softmax(axis=-2, name='pwm')(pwm_logits)

	#Optionally tile each PWM to sample from
	if use_samples :
		pwm_logits = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(pwm_logits)
	
	#Sample proper One-hot coded sequences from PWMs
	if validation_sample_mode == 'max' :
		sampled_pwm = Lambda(sample_pwm, name='pwm_sampler')(pwm_logits)
	elif validation_sample_mode == 'gumbel' :
		sampled_pwm = Lambda(sample_gumbel, name='pwm_sampler')(pwm_logits)
	elif validation_sample_mode == 'simple_sample' :
		sampled_pwm = Lambda(sample_pwm_simple, name='pwm_sampler')(pwm_logits)
	elif validation_sample_mode == 'sample_2' :
		sampled_pwm = Lambda(sample_2, name='pwm_sampler')(pwm_logits)
	elif validation_sample_mode == 'max_2' :
		sampled_pwm = Lambda(max_2, name='pwm_sampler')(pwm_logits)
	else :
		sampled_pwm = Lambda(sample_pwm_only, name='pwm_sampler')(pwm_logits)
	
	#PWM & Sampled One-hot custom transform function
	if pwm_transform_func is not None :
		pwm = Lambda(lambda pwm_seq: pwm_transform_func(pwm_seq))(pwm)
		sampled_pwm = Lambda(lambda pwm_seq: pwm_transform_func(pwm_seq))(sampled_pwm)
	
	#Optionally create sample axis
	if use_samples :
		sampled_pwm = Lambda(lambda x: K.reshape(x, (n_samples, n_sequences, seq_length, 20, 1)))(sampled_pwm)

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

