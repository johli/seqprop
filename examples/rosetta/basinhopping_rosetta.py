import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import basinhopping, OptimizeResult

import isolearn.keras as iso

from seqprop_rosetta_kl_helper import _get_kl_divergence_numpy, _get_smooth_kl_divergence_numpy, _get_smooth_circular_kl_divergence_numpy

#Simulated annealing (Basin hopping)

class IdentityEncoder(iso.SequenceEncoder) :
    
    def __init__(self, seq_len, channel_map) :
        super(IdentityEncoder, self).__init__('identity', (seq_len, len(channel_map)))
        
        self.seq_len = seq_len
        self.n_channels = len(channel_map)
        self.encode_map = channel_map
        self.decode_map = {
            nt: ix for ix, nt in self.encode_map.items()
        }
    
    def encode(self, seq) :
        encoding = np.zeros((self.seq_len, self.n_channels))
        
        for i in range(len(seq)) :
            if seq[i] in self.encode_map :
                channel_ix = self.encode_map[seq[i]]
                encoding[i, channel_ix] = 1.

        return encoding
    
    def encode_inplace(self, seq, encoding) :
        for i in range(len(seq)) :
            if seq[i] in self.encode_map :
                channel_ix = self.encode_map[seq[i]]
                encoding[i, channel_ix] = 1.
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        raise NotImplementError()
    
    def decode(self, encoding) :
        seq = ''
    
        for pos in range(0, encoding.shape[0]) :
            argmax_nt = np.argmax(encoding[pos, :])
            max_nt = np.max(encoding[pos, :])
            seq += self.decode_map[argmax_nt]

        return seq
    
    def decode_sparse(self, encoding_mat, row_index) :
        raise NotImplementError()


def get_step_func(predictor, sequence_template, acgt_encoder, n_swaps=1) :
    
    available_positions = [
        j for j in range(len(sequence_template)) if sequence_template[j] == '$'
    ]
    
    residues = list("ARNDCQEGHILKMFPSTWYV")
    
    available_nt_dict = {
        residue_ix : [residue_ix_2 for residue_ix_2 in range(len(residues)) if residue_ix_2 != residue_ix]
        for residue_ix in range(len(residues))
        #0 : [1, 2, 3],
        #1 : [0, 2, 3],
        #2 : [1, 0, 3],
        #3 : [1, 2, 0]
    }
        
    #_predict_func = get_predict_func(predictor, len(sequence_template))
    
    def _step_func(x, sequence_template=sequence_template, available_positions=available_positions, available_nt_dict=available_nt_dict, n_swaps=n_swaps) :
        
        onehot = np.expand_dims(np.expand_dims(x.reshape((len(sequence_template), 20)), axis=0), axis=-1)
        
        #Choose random position and nucleotide identity
        for swap_i in range(n_swaps) :
            rand_pos = np.random.choice(available_positions)

            curr_nt = np.argmax(onehot[0, rand_pos, :, 0])
            rand_nt = np.random.choice(available_nt_dict[curr_nt])

            #Swap nucleotides
            onehot[0, rand_pos, :, 0] = 0.
            onehot[0, rand_pos, rand_nt, 0] = 1.
        
        new_x = np.ravel(onehot)
        
        return new_x
    
    return _step_func

def get_predict_func(predictor, t_distos, msa_one_hot, seq_len) :
    
    def _predict_func(x, predictor=predictor, seq_len=seq_len, t_distos=t_distos, msa_one_hot=msa_one_hot) :
        
        td, tt, tp, to = t_distos
        
        onehot = np.expand_dims(x.reshape((seq_len, 20)), axis=0)
        
        msa_one_hot = np.expand_dims(onehot, axis=0)
        msa_one_hot = np.concatenate([msa_one_hot, np.zeros((1, 1, msa_one_hot.shape[2], 1))], axis=-1)
        
        p_dist, p_theta, p_phi, p_omega = predictor.predict(x=[onehot, msa_one_hot], batch_size=1)
        
        t_dist = np.clip(td, 1e-7, 1. - 1e-7)
        t_theta = np.clip(tt, 1e-7, 1. - 1e-7)
        t_phi = np.clip(tp, 1e-7, 1. - 1e-7)
        t_omega = np.clip(to, 1e-7, 1. - 1e-7)
        
        kl_dist, kl_theta, kl_phi, kl_omega = _get_kl_divergence_numpy(p_dist, p_theta, p_phi, p_omega, t_dist, t_theta, t_phi, t_omega)

        return kl_dist + kl_theta + kl_phi + kl_omega
    
    return _predict_func

def get_predict_func_smooth_kl(predictor, t_distos, msa_one_hot, seq_len) :
    
    def _predict_func_smooth_kl(x, predictor=predictor, seq_len=seq_len, t_distos=t_distos, msa_one_hot=msa_one_hot) :
        
        td, tt, tp, to = t_distos
        
        onehot = np.expand_dims(x.reshape((seq_len, 20)), axis=0)
        
        msa_one_hot = np.expand_dims(onehot, axis=0)
        msa_one_hot = np.concatenate([msa_one_hot, np.zeros((1, 1, msa_one_hot.shape[2], 1))], axis=-1)
        
        p_dist, p_theta, p_phi, p_omega = predictor.predict(x=[onehot, msa_one_hot], batch_size=1)
        
        t_dist = np.clip(td, 1e-7, 1. - 1e-7)
        t_theta = np.clip(tt, 1e-7, 1. - 1e-7)
        t_phi = np.clip(tp, 1e-7, 1. - 1e-7)
        t_omega = np.clip(to, 1e-7, 1. - 1e-7)
        
        kl_dist, kl_theta, kl_phi, kl_omega = _get_smooth_kl_divergence_numpy(p_dist, p_theta, p_phi, p_omega, t_dist, t_theta, t_phi, t_omega)

        return kl_dist + kl_theta + kl_phi + kl_omega
    
    return _predict_func_smooth_kl

def get_predict_func_smooth_circular_kl(predictor, t_distos, msa_one_hot, seq_len) :
    
    def _predict_func_smooth_circular_kl(x, predictor=predictor, seq_len=seq_len, t_distos=t_distos, msa_one_hot=msa_one_hot) :
        
        td, tt, tp, to = t_distos
        
        onehot = np.expand_dims(x.reshape((seq_len, 20)), axis=0)
        
        msa_one_hot = np.expand_dims(onehot, axis=0)
        msa_one_hot = np.concatenate([msa_one_hot, np.zeros((1, 1, msa_one_hot.shape[2], 1))], axis=-1)
        
        p_dist, p_theta, p_phi, p_omega = predictor.predict(x=[onehot, msa_one_hot], batch_size=1)
        
        t_dist = np.clip(td, 1e-7, 1. - 1e-7)
        t_theta = np.clip(tt, 1e-7, 1. - 1e-7)
        t_phi = np.clip(tp, 1e-7, 1. - 1e-7)
        t_omega = np.clip(to, 1e-7, 1. - 1e-7)
        
        kl_dist, kl_theta_sin, kl_theta_cos, kl_phi, kl_omega_sin, kl_omega_cos = _get_smooth_circular_kl_divergence_numpy(p_dist, p_theta, p_phi, p_omega, t_dist, t_theta, t_phi, t_omega)

        return kl_dist + kl_theta_sin + kl_theta_cos + kl_phi + kl_omega_sin + kl_omega_cos
    
    return _predict_func_smooth_circular_kl

def run_simulated_annealing(predictor, t_distos, msa_one_hot, sequence_template, acgt_encoder, n_iters=1000, n_iters_per_temperate=100, temperature_init=1.0, temperature_func=None, n_swaps=1, verbose=False) :
    
    if temperature_func is None :
        temperature_func = lambda t, curr_iter, t_init=temperature_init, total_iters=n_iters: t
    
    n_epochs = n_iters // n_iters_per_temperate
    
    predict_func = get_predict_func(predictor, t_distos, msa_one_hot, len(sequence_template))
    predict_func_smooth_kl = get_predict_func_smooth_kl(predictor, t_distos, msa_one_hot, len(sequence_template))
    predict_func_smooth_circular_kl = get_predict_func_smooth_circular_kl(predictor, t_distos, msa_one_hot, len(sequence_template))
    step_func = get_step_func(predictor, sequence_template, acgt_encoder, n_swaps=n_swaps)
    
    #Random initialization
    random_sequence = ''.join([
        sequence_template[j] if sequence_template[j] != '$' else np.random.choice(list("ARNDCQEGHILKMFPSTWYV"))
        for j in range(len(sequence_template))
    ])

    x0 = np.ravel(acgt_encoder.encode(random_sequence))
    
    x = x0
    temperature = temperature_init
    
    tracked_scores = [-predict_func(x)]
    tracked_scores_smooth = [-predict_func_smooth_kl(x)]
    tracked_scores_smooth_circular = [-predict_func_smooth_circular_kl(x)]
    for epoch_ix in range(n_epochs) :
        
        x_opt, f_opt = run_basinhopping(x, predict_func, step_func, n_iters=n_iters_per_temperate, temperature=temperature)
    
        score_opt = -f_opt
        tracked_scores.append(score_opt)
        tracked_scores_smooth.append(-predict_func_smooth_kl(x_opt))
        tracked_scores_smooth_circular.append(-predict_func_smooth_circular_kl(x_opt))
        
        if verbose :
            print("Iter " + str((epoch_ix + 1) * n_iters_per_temperate) + ", Temp = " + str(round(temperature, 4)) + ", Score = " + str(round(score_opt, 4)) + "...")

        x = x_opt
        temperature = temperature_func(temperature, (epoch_ix + 1) * n_iters_per_temperate)
    
    onehot_opt = np.expand_dims(np.expand_dims(x.reshape((len(sequence_template), 20)), axis=0), axis=-1)
    seq_opt = acgt_encoder.decode(onehot_opt[0, :, :, 0])
    
    return seq_opt, np.array(tracked_scores), np.array(tracked_scores_smooth), np.array(tracked_scores_smooth_circular)

def run_basinhopping(x, predict_func, step_func, n_iters=1000, temperature=1.0) :
    
    def _dummy_min_opt(fun, x0, args=(), **options) :
        return OptimizeResult(fun=fun(x0), x=x0, nit=0, nfev=0, success=True)
    
    minimizer_kwargs = {
        'method' : _dummy_min_opt,
        'options' : { 'maxiter' : 0 }
    }
    
    opt_res = basinhopping(predict_func, x, minimizer_kwargs=minimizer_kwargs, stepsize=None, niter=n_iters, T=temperature, take_step=step_func)
    
    return opt_res.x, opt_res.fun

def run_simulated_annealing_batch(saved_predictor, t_distos, msa_one_hot, sequence_template, acgt_encoder, n_sequences=1, n_iters=1000, n_iters_per_temperate=100, temperature_init=1.0, temperature_func=None, n_swaps=1, verbose=False) :
    
    if verbose :
        f = plt.figure(figsize=(6, 4))

        it_space = [0] + [(epoch_ix + 1) * n_iters_per_temperate for epoch_ix in range(n_iters // n_iters_per_temperate)]
        temp = temperature_init
        temp_space = [temp]
        for j in range(1, len(it_space)) :
            it = it_space[j]
            temp = temperature_func(temp, it)
            temp_space.append(temp)

        plt.plot(it_space, temp_space, linewidth=2, color='black', linestyle='-')

        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Temperature", fontsize=14)
        plt.title("Anneal schedule", fontsize=14)

        plt.xlim(0, np.max(it_space))

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.tight_layout()
        plt.show()
    
    optimized_seqs = []
    optimized_trajs = []
    optimized_trajs_smooth_kl = []
    optimized_trajs_smooth_circular_kl = []
    for sequence_ix in range(n_sequences) :

        seq, scores, scores_smooth_kl, scores_smooth_circular_kl = run_simulated_annealing(saved_predictor, t_distos, msa_one_hot, sequence_template, acgt_encoder, n_iters=n_iters, n_iters_per_temperate=n_iters_per_temperate, temperature_init=temperature_init, temperature_func=temperature_func, n_swaps=n_swaps, verbose=verbose)

        optimized_seqs.append(seq)
        optimized_trajs.append(scores.reshape(1, -1))
        optimized_trajs_smooth_kl.append(scores_smooth_kl.reshape(1, -1))
        optimized_trajs_smooth_circular_kl.append(scores_smooth_circular_kl.reshape(1, -1))

    optimized_trajs = np.concatenate(optimized_trajs, axis=0)
    optimized_trajs_smooth_kl = np.concatenate(optimized_trajs_smooth_kl, axis=0)
    optimized_trajs_smooth_circular_kl = np.concatenate(optimized_trajs_smooth_circular_kl, axis=0)
    
    print("[Basinhopping] Finished optimizing " + str(n_sequences) + " sequences. Final loss = " + str(np.mean(-optimized_trajs[:, -1])))
    
    return optimized_seqs, optimized_trajs, optimized_trajs_smooth_kl, optimized_trajs_smooth_circular_kl
