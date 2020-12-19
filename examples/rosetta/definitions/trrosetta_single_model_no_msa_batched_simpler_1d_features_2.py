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

# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
	beff = tf.reduce_sum(w)
	f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 0.005#1e-9
	h_i = tf.reduce_sum( -f_i * tf.math.log(f_i), axis=1)
	return tf.concat([f_i, h_i[:,None]], axis=1)

# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
	with tf.name_scope('reweight'):
		id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff
		id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
		id_mask = id_mtx > id_min
		w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
	return w

# shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty = 4.5):

	nr = tf.shape(msa1hot)[0]
	nc = tf.shape(msa1hot)[1]
	ns = tf.shape(msa1hot)[2]

	with tf.name_scope('covariance'):
		x = tf.reshape(msa1hot, (nr, nc * ns))
		num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
		mean = tf.reduce_sum(x * weights[:,None], axis=0, keepdims=True) / num_points
		x = (x - mean) * tf.sqrt(weights[:,None])
		cov = tf.matmul(tf.transpose(x), x)/num_points

	with tf.name_scope('inv_convariance'):
		cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(weights))
		inv_cov = tf.linalg.inv(cov_reg)
		
		x1 = tf.reshape(inv_cov,(nc, ns, nc, ns))
		x2 = tf.transpose(x1, [0,2,1,3])
		features = tf.reshape(x2, (nc, nc, ns * ns))
		
		x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(nc))
		apc = tf.reduce_sum(x3,0,keepdims=True) * tf.reduce_sum(x3,1,keepdims=True) / tf.reduce_sum(x3)
		contacts = (x3 - apc) * (1-tf.eye(nc))

	return tf.concat([features, contacts[:,:,None]], axis=2)

def keras_collect_features(inputs, wmin=0.8) :
	f1d_seq_batched, msa1hot_batched = inputs

	f1d_seq = f1d_seq_batched[0, ...]
	msa1hot = msa1hot_batched[0, ...]

	nrow = K.shape(msa1hot)[0]
	ncol = K.shape(msa1hot)[1]

	w = reweight(msa1hot, wmin)

	# 1D features
	f1d_pssm = msa2pssm(msa1hot, w)

	f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
	f1d = tf.expand_dims(f1d, axis=0)
	f1d = tf.reshape(f1d, [1,ncol,42])

	# 2D features
	f2d_dca = tf.cond(nrow>1, lambda: fast_dca(msa1hot, w), lambda: tf.zeros([ncol,ncol,442], tf.float32))
	f2d_dca = tf.expand_dims(f2d_dca, axis=0)

	f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]), 
					tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
					f2d_dca], axis=-1)
	f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])

	return f2d

#trRosetta Saved Model definition

def pssm_func(inputs, diag=0.0):
	x,y = inputs
	_,_,L,A = [tf.shape(y)[k] for k in range(4)]
	with tf.name_scope('1d_features'):
		# sequence
		x_i = x[0,:,:20]
		# pssm
		f_i = y[0,0, :, :]
		# entropy
		h_i = tf.zeros((L,1))
		#h_i = K.sum(-f_i * K.log(f_i + 1e-8), axis=-1, keepdims=True)
		# tile and combined 1D features
		feat_1D = tf.concat([x_i,f_i,h_i], axis=-1)
		feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
		feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

	with tf.name_scope('2d_features'):
		ic = diag * tf.eye(L*A)
		ic = tf.reshape(ic,(L,A,L,A))
		ic = tf.transpose(ic,(0,2,1,3))
		ic = tf.reshape(ic,(L,L,A*A))
		i0 = tf.zeros([L,L,1])
		feat_2D = tf.concat([ic,i0], axis=-1)

		feat = tf.concat([feat_1D_tile_A, feat_1D_tile_B, feat_2D],axis=-1)
		return tf.reshape(feat, [1,L,L,442+2*42])

def load_saved_predictor(model_path, msa_one_hot=None) :
	
	saved_model = load_model(model_path, custom_objects = {
		'InstanceNormalization' : InstanceNormalization,
		'reweight' : reweight,
		'wmin' : 0.8,
		'msa2pssm' : msa2pssm,
		'tf' : tf,
		'fast_dca' : fast_dca,
		'keras_collect_features' : pssm_func#keras_collect_features
	})
	#print(saved_model.summary())

	def _initialize_predictor_weights(predictor_model, saved_model=saved_model, model_path=model_path) :
		#Load pre-trained model
		#print(saved_model.summary())
		#predictor_model.load_weights(model_path, by_name=True)
		print("No weights copied.")

	def _load_predictor_func(sequence_input, saved_model=saved_model) :
		
		msa_one_hot_var = Lambda(lambda x: K.concatenate([K.expand_dims(x, axis=1), K.zeros((K.shape(x)[0], 1, K.shape(x)[1], 1, 1))], axis=-2))(sequence_input)
		
		p_dist, p_theta, p_phi, p_omega = Lambda(lambda x: saved_model([x[0][..., 0], x[1][..., 0]]))([sequence_input, msa_one_hot_var])#saved_model([sequence_input, msa_one_hot_var])

		p_dist_clipped = Lambda(lambda x: K.clip(x, K.epsilon(), 1. - K.epsilon()))(p_dist)
		p_theta_clipped = Lambda(lambda x: K.clip(x, K.epsilon(), 1. - K.epsilon()))(p_theta)
		p_phi_clipped = Lambda(lambda x: K.clip(x, K.epsilon(), 1. - K.epsilon()))(p_phi)
		p_omega_clipped = Lambda(lambda x: K.clip(x, K.epsilon(), 1. - K.epsilon()))(p_omega)
		
		predictor_inputs = []
		predictor_outputs = [p_dist_clipped, p_theta_clipped, p_phi_clipped, p_omega_clipped]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func
