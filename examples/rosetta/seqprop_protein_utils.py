import string
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

def one_hot_encode_msa(msa, ns=21) :
	
	one_hot = np.zeros((msa.shape[0], msa.shape[1], ns))
	for i in range(msa.shape[0]) :
		for j in range(msa.shape[1]) :
			one_hot[i, j, int(msa[i, j])] = 1.
	
	return one_hot

def parse_a3m(filename):
	seqs = []
	table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

	# read file line by line
	for line in open(filename,"r"):
		# skip labels
		if line[0] != '>':
			# remove lowercase letters and right whitespaces
			seqs.append(line.rstrip().translate(table))

	# convert letters into numbers
	alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
	msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
	for i in range(alphabet.shape[0]):
		msa[msa == alphabet[i]] = i

	# treat all unknown characters as gaps
	msa[msa > 20] = 20

	return msa

# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
	beff = tf.reduce_sum(w)
	f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
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

def letterAt_protein(letter, x, y, yscale=1, ax=None, color='black', alpha=1.0):

	#fp = FontProperties(family="Arial", weight="bold")
	#fp = FontProperties(family="Ubuntu", weight="bold")
	fp = FontProperties(family="DejaVu Sans", weight="bold")
	
	globscale = 1.35
	LETTERS = {"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
			   "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
			   "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
			   "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
			   
			   "L" : TextPath((-0.35, 0), "L", size=1, prop=fp),
			   "M" : TextPath((-0.35, 0), "M", size=1, prop=fp),
			   "F" : TextPath((-0.35, 0), "F", size=1, prop=fp),
			   "W" : TextPath((-0.35, 0), "W", size=1, prop=fp),
			   "K" : TextPath((-0.35, 0), "K", size=1, prop=fp),
			   "Q" : TextPath((-0.35, 0), "Q", size=1, prop=fp),
			   "E" : TextPath((-0.35, 0), "E", size=1, prop=fp),
			   "S" : TextPath((-0.35, 0), "S", size=1, prop=fp),
			   "P" : TextPath((-0.35, 0), "P", size=1, prop=fp),
			   "V" : TextPath((-0.35, 0), "V", size=1, prop=fp),
			   "I" : TextPath((-0.35, 0), "I", size=1, prop=fp),
			   "Y" : TextPath((-0.35, 0), "Y", size=1, prop=fp),
			   "H" : TextPath((-0.35, 0), "H", size=1, prop=fp),
			   "R" : TextPath((-0.35, 0), "R", size=1, prop=fp),
			   "N" : TextPath((-0.35, 0), "N", size=1, prop=fp),
			   "D" : TextPath((-0.35, 0), "D", size=1, prop=fp),
			   "U" : TextPath((-0.35, 0), "U", size=1, prop=fp),
			   "!" : TextPath((-0.35, 0), "!", size=1, prop=fp),
			   
			   "UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
			   "DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
			   "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
			   "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
			   ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}

	
	if letter in LETTERS :
		text = LETTERS[letter]
	else :
		text = TextPath((-0.35, 0), letter, size=1, prop=fp)
	
	chosen_color = color
	
	if chosen_color is None :
		chosen_color = 'black'
		if letter in ['A', 'I', 'L', 'M', 'F', 'W', 'V'] : #Hydrophobic
			chosen_color = 'blue'
		elif letter in ['K' ,'R'] : #Positive charge
			chosen_color = 'red'
		elif letter in ['E', 'D'] : #Negative charge
			chosen_color = 'magenta'
		elif letter in ['N', 'Q', 'S', 'T'] : #Polar
			chosen_color = 'green'
		elif letter in ['C'] : #Cysteines
			chosen_color = 'pink'
		elif letter in ['G'] : #Glycines
			chosen_color = 'orange'
		elif letter in ['P'] : #Prolines
			chosen_color = 'yellow'
		elif letter in ['H', 'Y'] : #Aromatic
			chosen_color = 'cyan'

	t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
		mpl.transforms.Affine2D().translate(x,y) + ax.transData
	p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
	if ax != None:
		ax.add_artist(p)
	return p


def plot_protein_logo(inv_residue_map, pwm, meas_str, score, sequence_template=None, figsize=(12, 3), width_ratios=[1, 7], logo_height=1.0, plot_start=0, plot_end=164) :

	#Slice according to seq trim index
	pwm = pwm[plot_start: plot_end, :]
	sequence_template = sequence_template[plot_start: plot_end]

	pwm += 0.0001
	for j in range(0, pwm.shape[0]) :
		pwm[j, :] /= np.sum(pwm[j, :])

	entropy = np.zeros(pwm.shape)
	entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
	entropy = np.sum(entropy, axis=1)
	conservation = np.log2(len(inv_residue_map)) - entropy#2 - entropy

	fig = plt.figure(figsize=figsize)

	gs = gridspec.GridSpec(1, 2, width_ratios=[width_ratios[0], width_ratios[-1]])

	ax2 = plt.subplot(gs[0])
	ax3 = plt.subplot(gs[1])

	plt.sca(ax2)
	plt.axis('off')


	annot_text = '\n' + meas_str + ' = ' + str(round(score, 4))

	ax2.text(0.99, 0.5, annot_text, horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, color='black', fontsize=12, weight="bold")

	height_base = (1.0 - logo_height) / 2.

	for j in range(0, pwm.shape[0]) :
		sort_index = np.argsort(pwm[j, :])

		for ii in range(0, len(inv_residue_map)) :
			i = sort_index[ii]

			nt_prob = pwm[j, i] * conservation[j]

			nt = inv_residue_map[i]

			color = None
			if sequence_template[j] != '$' :
				color = 'black'

			if ii == 0 :
				letterAt_protein(nt, j + 0.5, height_base, nt_prob * logo_height, ax3, color=color)
			else :
				prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
				letterAt_protein(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax3, color=color)

	plt.sca(ax3)

	plt.xlim((0, plot_end - plot_start))
	plt.ylim((0, np.log2(len(inv_residue_map))))
	plt.xticks([], [])
	plt.yticks([], [])
	plt.axis('off')
	ax3.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)


	for axis in fig.axes :
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)

	plt.tight_layout()

	plt.show()