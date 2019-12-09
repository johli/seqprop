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
	"""Instance normalization layer.
	Normalize the activations of the previous layer at each step,
	i.e. applies a transformation that maintains the mean activation
	close to 0 and the activation standard deviation close to 1.
	# Arguments
		axis: Integer, the axis that should be normalized
			(typically the features axis).
			For instance, after a `Conv2D` layer with
			`data_format="channels_first"`,
			set `axis=1` in `InstanceNormalization`.
			Setting `axis=None` will normalize all values in each
			instance of the batch.
			Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
		epsilon: Small float added to variance to avoid dividing by zero.
		center: If True, add offset of `beta` to normalized tensor.
			If False, `beta` is ignored.
		scale: If True, multiply by `gamma`.
			If False, `gamma` is not used.
			When the next layer is linear (also e.g. `nn.relu`),
			this can be disabled since the scaling
			will be done by the next layer.
		beta_initializer: Initializer for the beta weight.
		gamma_initializer: Initializer for the gamma weight.
		beta_regularizer: Optional regularizer for the beta weight.
		gamma_regularizer: Optional regularizer for the gamma weight.
		beta_constraint: Optional constraint for the beta weight.
		gamma_constraint: Optional constraint for the gamma weight.
	# Input shape
		Arbitrary. Use the keyword argument `input_shape`
		(tuple of integers, does not include the samples axis)
		when using this layer as the first layer in a Sequential model.
	# Output shape
		Same shape as input.
	# References
		- [Layer Normalization](https://arxiv.org/abs/1607.06450)
		- [Instance Normalization: The Missing Ingredient for Fast Stylization](
		https://arxiv.org/abs/1607.08022)
	"""
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

#PWM Masking and Sampling helper functions

def mask_pwm(inputs) :
	pwm, onehot_template, onehot_mask = inputs

	return pwm * onehot_mask + onehot_template

def sample_pwm_only(pwm_logits) :
	n_sequences = pwm_logits.get_shape().as_list()[0]
	seq_length = pwm_logits.get_shape().as_list()[1]
	
	flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, 4))
	sampled_pwm = st_sampled_softmax(flat_pwm)

	return K.reshape(sampled_pwm, (n_sequences, seq_length, 4, 1))

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

	onehot_logits = generator.get_layer('policy_pwm').get_weights()[0].reshape((len(init_sequences), len(init_sequences[0]), 4, 1))

	on_logit = np.log(p_init / (1. - p_init))

	p_off = (1. - p_init) / 3.
	off_logit = np.log(p_off / (1. - p_off))

	for i in range(len(init_sequences)) :
		init_sequence = init_sequences[i]

		for j in range(len(init_sequence)) :
			nt_ix = -1
			if init_sequence[j] == 'A' :
				nt_ix = 0
			elif init_sequence[j] == 'C' :
				nt_ix = 1
			elif init_sequence[j] == 'G' :
				nt_ix = 2
			elif init_sequence[j] == 'T' :
				nt_ix = 3

			onehot_logits[i, j, :, :] = off_logit
			if nt_ix != -1 :
				onehot_logits[i, j, nt_ix, :] = on_logit

	generator.get_layer('policy_pwm').set_weights([onehot_logits.reshape(1, -1)])



#SeqProp Generator Model definitions

#Generator that samples a single one-hot sequence per trainable PWM
def build_generator(seq_length, n_sequences=1, n_samples=None, sequence_templates=None, init_sequences=None, p_init=0.5, batch_normalize_pwm=False, pwm_transform_func=None, validation_sample_mode='max', master_generator=None) :

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

	if master_generator is not None :
		dense_seq_layer = master_generator.get_layer('policy_pwm')
	
	#Initialize Templating and Masking Lambda layer
	masking_layer = Lambda(mask_pwm, output_shape = (seq_length, 4, 1), name='masking_layer')
	
	#Get Template, Mask and Trainable PWM logits
	onehot_template = reshape_layer(onehot_template_dense(ones_input))
	onehot_mask = reshape_layer(onehot_mask_dense(ones_input))
	onehot_logits = reshape_layer(dense_seq_layer(ones_input))

	#Batch Normalize PWM Logits
	if batch_normalize_pwm :
		pwm_norm_layer = InstanceNormalization(axis=-2, name='policy_batch_norm')
		if master_generator is not None :
			pwm_norm_layer = master_generator.get_layer('policy_batch_norm')
		onehot_logits = pwm_norm_layer(onehot_logits)
	
	#Add Template and Multiply Mask
	pwm_logits = masking_layer([onehot_logits, onehot_template, onehot_mask])
	
	#Get PWM from logits
	pwm = Softmax(axis=-2, name='pwm')(pwm_logits)

	#Optionally tile each PWM to sample from
	if use_samples :
		pwm_logits = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(pwm_logits)
	
	#Sample proper One-hot coded sequences from PWMs
	if validation_sample_mode == 'max' :
		sampled_pwm = Lambda(sample_pwm, name='pwm_sampler')(pwm_logits)
	else :
		sampled_pwm = Lambda(sample_pwm_only, name='pwm_sampler')(pwm_logits)
	
	#PWM & Sampled One-hot custom transform function
	if pwm_transform_func is not None :
		pwm = Lambda(lambda pwm_seq: pwm_transform_func(pwm_seq))(pwm)
		sampled_pwm = Lambda(lambda pwm_seq: pwm_transform_func(pwm_seq))(sampled_pwm)
	
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
		initialize_sequences(generator_model, init_sequences, p_init)

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
