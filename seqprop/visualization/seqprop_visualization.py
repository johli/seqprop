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

import keras
from keras.callbacks import Callback
from keras import backend as K


def letterAt(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

	#fp = FontProperties(family="Arial", weight="bold")
	fp = FontProperties(family="Ubuntu", weight="bold")
	globscale = 1.35
	LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
				"G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
				"A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
				"C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
				"UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
				"DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
				"(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
				"." : TextPath((-0.125, 0), "-", size=1, prop=fp),
				")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}
	COLOR_SCHEME = {'G': 'orange', 
					'A': 'red', 
					'C': 'blue', 
					'T': 'darkgreen',
					'UP': 'green', 
					'DN': 'red',
					'(': 'black',
					'.': 'black', 
					')': 'black'}


	text = LETTERS[letter]

	chosen_color = COLOR_SCHEME[letter]
	if color is not None :
		chosen_color = color

	t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
		mpl.transforms.Affine2D().translate(x,y) + ax.transData
	p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
	if ax != None:
		ax.add_artist(p)
	return p

def plot_seqprop_logo(pwm, iso_pred, cut_pred, cse_start_pos=70, annotate_peaks=None, sequence_template=None, figsize=(12, 3), width_ratios=[1, 7], logo_height=1.0, usage_unit='log', plot_start=0, plot_end=164, save_figs=False, fig_name=None, fig_dpi=300) :

	n_samples = pwm.shape[0]
	
	#Slice according to seq trim index
	pwm = pwm[:, plot_start: plot_end, :]
	cut_pred = cut_pred[:, plot_start: plot_end]
	sequence_template = sequence_template[plot_start: plot_end]
	
	iso_pred = np.mean(iso_pred, axis=0)
	cut_pred = np.mean(cut_pred, axis=0)
	pwm = np.sum(pwm, axis=0)
	
	pwm += 0.0001
	for j in range(0, pwm.shape[0]) :
		pwm[j, :] /= np.sum(pwm[j, :])
	
	entropy = np.zeros(pwm.shape)
	entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
	entropy = np.sum(entropy, axis=1)
	conservation = 2 - entropy

	fig = plt.figure(figsize=figsize)
	
	gs = gridspec.GridSpec(2, 2, width_ratios=[width_ratios[0], width_ratios[-1]], height_ratios=[1, 1])
	
	ax0 = plt.subplot(gs[0, 0])
	ax1 = plt.subplot(gs[0, 1])
	ax2 = plt.subplot(gs[1, 0])
	ax3 = plt.subplot(gs[1, 1])
	
	plt.sca(ax0)
	plt.axis('off')
	plt.sca(ax2)
	plt.axis('off')
	
	
	annot_text = 'Samples = ' + str(int(n_samples))
	if usage_unit == 'log' :
		annot_text += '\nUsage = ' + str(round(np.log(iso_pred[0] / (1. - iso_pred[0])), 4))
	else :
		annot_text += '\nUsage = ' + str(round(iso_pred[0], 4))
		
	ax2.text(0.99, 0.5, annot_text, horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, color='black', fontsize=12, weight="bold")

	l2, = ax1.plot(np.arange(plot_end - plot_start), cut_pred, linewidth=3, linestyle='-', label='Predicted', color='red', alpha=0.7)
	
	if annotate_peaks is not None :
		objective_pos = 0
		if annotate_peaks == 'max' :
			objective_pos = np.argmax(cut_pred)
		else :
			objective_pos = annotate_peaks - plot_start
		
		text_x, text_y, ha = -30, -5, 'right'
		if objective_pos < 30 :
			text_x, text_y, ha = 30, -5, 'left'

		annot_text = '(CSE+' + str(objective_pos + plot_start - (cse_start_pos + 6) + 0) + ') ' + str(int(round(cut_pred[objective_pos] * 100, 0))) + '% Cleavage'
		ax1.annotate(annot_text, xy=(objective_pos, cut_pred[objective_pos]), xycoords='data', xytext=(text_x, text_y), ha=ha, fontsize=10, weight="bold", color='black', textcoords='offset points', arrowprops=dict(connectionstyle="arc3,rad=-.1", headlength=8, headwidth=8, shrink=0.15, width=1.5, color='black'))
	
	plt.sca(ax1)

	plt.xlim((0, plot_end - plot_start))
	#plt.ylim((0, 2))
	plt.xticks([], [])
	plt.yticks([], [])
	plt.legend(handles=[l2], fontsize=12, prop=dict(weight='bold'), frameon=False)
	plt.axis('off')
	
	height_base = (1.0 - logo_height) / 2.

	for j in range(0, pwm.shape[0]) :
		sort_index = np.argsort(pwm[j, :])

		for ii in range(0, 4) :
			i = sort_index[ii]

			nt_prob = pwm[j, i] * conservation[j]

			nt = ''
			if i == 0 :
				nt = 'A'
			elif i == 1 :
				nt = 'C'
			elif i == 2 :
				nt = 'G'
			elif i == 3 :
				nt = 'T'

			color = None
			if sequence_template[j] != 'N' :
				color = 'black'

			if ii == 0 :
				letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, ax3, color=color)
			else :
				prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
				letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax3, color=color)

	plt.sca(ax3)

	plt.xlim((0, plot_end - plot_start))
	plt.ylim((0, 2))
	plt.xticks([], [])
	plt.yticks([], [])
	plt.axis('off')
	ax3.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)


	for axis in fig.axes :
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)
	
	plt.tight_layout()
	
	if save_figs :
		plt.savefig(fig_name + '.png', transparent=True, dpi=fig_dpi)
		plt.savefig(fig_name + '.svg')
		plt.savefig(fig_name + '.eps')
	
	plt.show()
	#plt.close()

#Sequence optimization monitor during training
class SeqPropMonitor(Callback):
	def __init__(self, predictor, plot_every_epoch=False, track_every_step=False, cse_start_pos=70, isoform_start=80, isoform_end=110, dedicated_isoform_pred=False, pwm_start=0, pwm_end=100, sequence_template='', plot_pwm_indices=[]) :
		self.predictor = predictor
		self.plot_every_epoch = plot_every_epoch
		self.track_every_step = track_every_step
		self.plot_pwm_indices = plot_pwm_indices
		self.isoform_start = isoform_start
		self.isoform_end = isoform_end
		self.pwm_start = pwm_start
		self.pwm_end = pwm_end
		self.sequence_template = sequence_template
		self.dedicated_isoform_pred = dedicated_isoform_pred
		self.cse_start_pos = cse_start_pos
		
		self.iso_history = []
		self.entropy_history = []
		self.nt_swap_history = []
		self.prev_optimized_pwm = None
		
		self.n_epochs = 0
		
		pred_bundle = self.predictor.predict(x=None, steps=1)
		if len(pred_bundle) == 7 :
			_, optimized_pwm, _, iso_pred, cut_pred, _, _ = pred_bundle
		else :
			_, optimized_pwm, _, iso_pred, cut_pred, _, _, _ = pred_bundle
		
		if len(cut_pred.shape) > 2 :
			iso_pred = iso_pred[0, :, :]
			cut_pred = cut_pred[0, :, :]
		
		if not self.dedicated_isoform_pred :
			iso_pred = np.expand_dims(np.sum(cut_pred[:, isoform_start:isoform_end], axis=-1), axis=-1)
		
		#Track metrics
		self._track_iso_history(iso_pred)
		self._track_entropy_history(optimized_pwm)
		
		self.prev_optimized_pwm = optimized_pwm
		self.nt_swap_history.append(np.zeros((optimized_pwm.shape[0], 1)))
	
	def _track_iso_history(self, iso_pred) :
		self.iso_history.append(np.log(iso_pred / (1. - iso_pred)))
	
	def _track_entropy_history(self, optimized_pwm) :
		pwm_section = optimized_pwm[:, self.pwm_start:self.pwm_end, :, :]
		entropy = pwm_section * -np.log(np.clip(pwm_section, 10**(-6), 1. - 10**(-6))) / np.log(2.0)
		entropy = np.sum(entropy, axis=(2, 3))
		conservation = 2.0 - entropy
		mean_bits = np.expand_dims(np.mean(conservation, axis=-1), axis=-1)
		self.entropy_history.append(mean_bits)
	
	def _track_nt_swap_history(self, optimized_pwm) :
		nt_swaps = np.zeros((optimized_pwm.shape[0], 1))
		nt_swaps[:, 0] = self.nt_swap_history[-1][:, 0]
		
		for i in range(optimized_pwm.shape[0]) :
			for j in range(self.pwm_start, self.pwm_end) :
				curr_max_nt = np.argmax(optimized_pwm[i, j, :, 0])
				prev_max_nt = np.argmax(self.prev_optimized_pwm[i, j, :, 0])
				
				if curr_max_nt != prev_max_nt :
					nt_swaps[i, 0] += 1
		
		self.nt_swap_history.append(nt_swaps)
	
	def _plot_metric_on_axis(self, ax, epoch_history, metric_label) :
		epoch_mat = np.concatenate(epoch_history, axis=-1)
		for i in range(epoch_mat.shape[0]) :
			ax.plot(np.arange(epoch_mat.shape[1]), epoch_mat[i, :], linewidth=2)

		plt.sca(ax)
		plt.title(metric_label, fontsize=14)
		plt.xlabel("Epoch", fontsize=14)
		plt.ylabel(metric_label, fontsize=14)

		#if epoch_mat.shape[1] <= 15 :
		#	plt.xticks(np.arange(epoch_mat.shape[1]), np.arange(epoch_mat.shape[1]), fontsize=14)
		#else :
		#	plt.xticks([0, epoch_mat.shape[1] - 1], [0, epoch_mat.shape[1] - 1], fontsize=14)
		plt.xticks([0, epoch_mat.shape[1] - 1], [0, self.n_epochs], fontsize=14)
		
		plt.yticks(fontsize=14)

		plt.xlim(0, epoch_mat.shape[1] - 1)
		plt.ylim(np.min(epoch_mat) - 0.02 * np.min(epoch_mat) * np.sign(np.min(epoch_mat)), np.max(epoch_mat) + 0.02 * np.max(epoch_mat) * np.sign(np.max(epoch_mat)))
	
	def on_batch_end(self, batch, logs={}) :
		
		if self.track_every_step :
			pred_bundle = self.predictor.predict(x=None, steps=1)
			if len(pred_bundle) == 7 :
				_, optimized_pwm, _, iso_pred, cut_pred, _, _ = pred_bundle
			else :
				_, optimized_pwm, _, iso_pred, cut_pred, _, _, _ = pred_bundle

			if len(cut_pred.shape) > 2 :
				iso_pred = iso_pred[0, :, :]
				cut_pred = cut_pred[0, :, :]

			if not self.dedicated_isoform_pred :
				iso_pred = np.expand_dims(np.sum(cut_pred[:, self.isoform_start:self.isoform_end], axis=-1), axis=-1)
		
			#Track measures
			self._track_iso_history(iso_pred)
			self._track_entropy_history(optimized_pwm)
			self._track_nt_swap_history(optimized_pwm)

			#Cache previous pwms
			self.prev_optimized_pwm = optimized_pwm
	
	def on_epoch_end(self, epoch, logs={}) :
		self.n_epochs += 1
		
		pred_bundle = self.predictor.predict(x=None, steps=1)
		if len(pred_bundle) == 7 :
			_, optimized_pwm, _, iso_pred, cut_pred, _, _ = pred_bundle
		else :
			_, optimized_pwm, _, iso_pred, cut_pred, _, _, _ = pred_bundle
		
		if len(cut_pred.shape) > 2 :
			iso_pred = iso_pred[0, :, :]
			cut_pred = cut_pred[0, :, :]
		
		if not self.dedicated_isoform_pred :
			iso_pred = np.expand_dims(np.sum(cut_pred[:, self.isoform_start:self.isoform_end], axis=-1), axis=-1)
		
		if not self.track_every_step :
		
			#Track measures
			self._track_iso_history(iso_pred)
			self._track_entropy_history(optimized_pwm)
			self._track_nt_swap_history(optimized_pwm)

			#Cache previous pwms
			self.prev_optimized_pwm = optimized_pwm
		
		if self.plot_every_epoch :
			
			f, ax = plt.subplots(1, 3, figsize=(9, 2.5))

			#Plot isoform usage
			self._plot_metric_on_axis(ax[0], self.iso_history, "Isoform (log odds)")

			#Plot pwm entropy
			self._plot_metric_on_axis(ax[1], self.entropy_history, "PWM Entropy (bits)")

			#Plot consensus nucleotide swap
			self._plot_metric_on_axis(ax[2], self.nt_swap_history, "Nucleotide Swaps")

			plt.tight_layout()
			plt.show()

			#Plot chosen PWM sequence logos
			figsize=(9, 1.5)

			plot_start = self.pwm_start
			plot_end = self.pwm_end

			for pwm_index in self.plot_pwm_indices :
				pwm = np.expand_dims(optimized_pwm[pwm_index, :, :, 0], axis=0)

				iso = np.expand_dims(iso_pred[pwm_index, :], axis=0)
				cut = np.expand_dims(cut_pred[pwm_index, :], axis=0)

				plot_seqprop_logo(pwm, iso, cut, cse_start_pos=self.cse_start_pos, annotate_peaks='max', sequence_template=self.sequence_template, figsize=figsize, width_ratios=[1, 8], logo_height=0.8, usage_unit='log', plot_start=plot_start, plot_end=plot_end)

	def on_train_end(self, logs={}) :
		
		if not self.plot_every_epoch :
			pred_bundle = self.predictor.predict(x=None, steps=1)
			if len(pred_bundle) == 7 :
				_, optimized_pwm, _, iso_pred, cut_pred, _, _ = pred_bundle
			else :
				_, optimized_pwm, _, iso_pred, cut_pred, _, _, _ = pred_bundle

			if len(cut_pred.shape) > 2 :
				iso_pred = iso_pred[0, :, :]
				cut_pred = cut_pred[0, :, :]

			if not self.dedicated_isoform_pred :
				iso_pred = np.expand_dims(np.sum(cut_pred[:, self.isoform_start:self.isoform_end], axis=-1), axis=-1)


			f, ax = plt.subplots(1, 3, figsize=(9, 2.5))

			#Plot isoform usage
			self._plot_metric_on_axis(ax[0], self.iso_history, "Isoform (log odds)")

			#Plot pwm entropy
			self._plot_metric_on_axis(ax[1], self.entropy_history, "PWM Entropy (bits)")

			#Plot consensus nucleotide swap
			self._plot_metric_on_axis(ax[2], self.nt_swap_history, "Nucleotide Swaps")

			plt.tight_layout()
			plt.show()

			#Plot chosen PWM sequence logos
			figsize=(9, 1.5)

			plot_start = self.pwm_start
			plot_end = self.pwm_end

			for pwm_index in self.plot_pwm_indices :
				pwm = np.expand_dims(optimized_pwm[pwm_index, :, :, 0], axis=0)

				iso = np.expand_dims(iso_pred[pwm_index, :], axis=0)
				cut = np.expand_dims(cut_pred[pwm_index, :], axis=0)

				plot_seqprop_logo(pwm, iso, cut, cse_start_pos=self.cse_start_pos, annotate_peaks='max', sequence_template=self.sequence_template, figsize=figsize, width_ratios=[1, 8], logo_height=0.8, usage_unit='log', plot_start=plot_start, plot_end=plot_end)
