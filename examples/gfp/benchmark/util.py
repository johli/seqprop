import itertools

import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from collections import Counter

from losses import get_gaussian_nll, summed_categorical_crossentropy, zero_loss, get_gaussian_nll_for_log_pred, identity_loss
from seqtools import SequenceTools
from vae import SimpleSupervisedVAE, SimpleVAE


AA = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v']
AA_IDX = {AA[i]:i for i in range(len(AA))}

BLOSUM = np.array([
[3.9029,0.6127,0.5883,0.5446,0.8680,0.7568,0.7413,1.0569,0.5694,0.6325,0.6019,0.7754,0.7232,0.4649,0.7541,1.4721,0.9844,0.4165,0.5426,0.9365],
[0.6127,6.6656,0.8586,0.5732,0.3089,1.4058,0.9608,0.4500,0.9170,0.3548,0.4739,2.0768,0.6226,0.3807,0.4815,0.7672,0.6778,0.3951,0.5560,0.4201],
[0.5883,0.8586,7.0941,1.5539,0.3978,1.0006,0.9113,0.8637,1.2220,0.3279,0.3100,0.9398,0.4745,0.3543,0.4999,1.2315,0.9842,0.2778,0.4860,0.3690],
[0.5446,0.5732,1.5539,7.3979,0.3015,0.8971,1.6878,0.6343,0.6786,0.3390,0.2866,0.7841,0.3465,0.2990,0.5987,0.9135,0.6948,0.2321,0.3457,0.3365],
[0.8680,0.3089,0.3978,0.3015,19.5766,0.3658,0.2859,0.4204,0.3550,0.6535,0.6423,0.3491,0.6114,0.4390,0.3796,0.7384,0.7406,0.4500,0.4342,0.7558],
[0.7568,1.4058,1.0006,0.8971,0.3658,6.2444,1.9017,0.5386,1.1680,0.3829,0.4773,1.5543,0.8643,0.3340,0.6413,0.9656,0.7913,0.5094,0.6111,0.4668],
[0.7413,0.9608,0.9113,1.6878,0.2859,1.9017,5.4695,0.4813,0.9600,0.3305,0.3729,1.3083,0.5003,0.3307,0.6792,0.9504,0.7414,0.3743,0.4965,0.4289],
[1.0569,0.4500,0.8637,0.6343,0.4204,0.5386,0.4813,6.8763,0.4930,0.2750,0.2845,0.5889,0.3955,0.3406,0.4774,0.9036,0.5793,0.4217,0.3487,0.3370],
[0.5694,0.9170,1.2220,0.6786,0.3550,1.1680,0.9600,0.4930,13.5060,0.3263,0.3807,0.7789,0.5841,0.6520,0.4729,0.7367,0.5575,0.4441,1.7979,0.3394],
[0.6325,0.3548,0.3279,0.3390,0.6535,0.3829,0.3305,0.2750,0.3263,3.9979,1.6944,0.3964,1.4777,0.9458,0.3847,0.4432,0.7798,0.4089,0.6304,2.4175],
[0.6019,0.4739,0.3100,0.2866,0.6423,0.4773,0.3729,0.2845,0.3807,1.6944,3.7966,0.4283,1.9943,1.1546,0.3711,0.4289,0.6603,0.5680,0.6921,1.3142],
[0.7754,2.0768,0.9398,0.7841,0.3491,1.5543,1.3083,0.5889,0.7789,0.3964,0.4283,4.7643,0.6253,0.3440,0.7038,0.9319,0.7929,0.3589,0.5322,0.4565],
[0.7232,0.6226,0.4745,0.3465,0.6114,0.8643,0.5003,0.3955,0.5841,1.4777,1.9943,0.6253,6.4815,1.0044,0.4239,0.5986,0.7938,0.6103,0.7084,1.2689],
[0.4649,0.3807,0.3543,0.2990,0.4390,0.3340,0.3307,0.3406,0.6520,0.9458,1.1546,0.3440,1.0044,8.1288,0.2874,0.4400,0.4817,1.3744,2.7694,0.7451],
[0.7541,0.4815,0.4999,0.5987,0.3796,0.6413,0.6792,0.4774,0.4729,0.3847,0.3711,0.7038,0.4239,0.2874,12.8375,0.7555,0.6889,0.2818,0.3635,0.4431],
[1.4721,0.7672,1.2315,0.9135,0.7384,0.9656,0.9504,0.9036,0.7367,0.4432,0.4289,0.9319,0.5986,0.4400,0.7555,3.8428,1.6139,0.3853,0.5575,0.5652],
[0.9844,0.6778,0.9842,0.6948,0.7406,0.7913,0.7414,0.5793,0.5575,0.7798,0.6603,0.7929,0.7938,0.4817,0.6889,1.6139,4.8321,0.4309,0.5732,0.9809],
[0.4165,0.3951,0.2778,0.2321,0.4500,0.5094,0.3743,0.4217,0.4441,0.4089,0.5680,0.3589,0.6103,1.3744,0.2818,0.3853,0.4309,38.1078,2.1098,0.3745],
[0.5426,0.5560,0.4860,0.3457,0.4342,0.6111,0.4965,0.3487,1.7979,0.6304,0.6921,0.5322,0.7084,2.7694,0.3635,0.5575,0.5732,2.1098,9.8322,0.6580],
[0.9365,0.4201,0.3690,0.3365,0.7558,0.4668,0.4289,0.3370,0.3394,2.4175,1.3142,0.4565,1.2689,0.7451,0.4431,0.5652,0.9809,0.3745,0.6580,3.6922]]
)



def build_pred_vae_model(latent_dim, n_tokens=4, seq_length=33, enc1_units=50,
                         eps_std=1., pred_var=0.1,
                         learn_uncertainty=False):
    model = SimpleSupervisedVAE(input_shape=(seq_length, n_tokens,),
                                latent_dim=latent_dim,
                                pred_dim=1,
                                pred_var=pred_var,
                                learn_uncertainty=learn_uncertainty)

    # set encoder layers:
    model.encoderLayers_ = [
        Dense(units=enc1_units, activation='elu', name='e2'),
    ]

    # set decoder layers:
    model.decoderLayers_ = [
        Dense(units=enc1_units, activation='elu', name='d1'),
        Dense(units=n_tokens * seq_length, name='d3'),
        Reshape((seq_length, n_tokens), name='d4'),
        Dense(units=n_tokens, activation='softmax', name='d5'),
    ]

    # set predictor layers:
    model.predictorLayers_ = [
        Dense(units=20, activation='elu', name='p1'),
    ]

    # build models:
    kl_scale = K.variable(1.)
    model.build_encoder()
    model.build_decoder(decode_activation='softmax')
    model.build_predictor()
    model.build_vae(epsilon_std=eps_std, kl_scale=kl_scale)

    y_var = K.exp(model.vae_.outputs[3])

    losses = [summed_categorical_crossentropy,
              identity_loss,
              get_gaussian_nll(y_var),
              zero_loss]

    model.compile(optimizer='adam',
                  loss=losses,
                  metrics=['mse'])
    return model


def build_pred_model(n_tokens=4, seq_length=33, enc1_units=50, pred_var=0.1):
    x = Input(shape=(seq_length, n_tokens))
    h = Flatten()(x)
    h = Dense(enc1_units, activation='elu')(h)
    h = Dense(enc1_units, activation='elu')(h)
    out = Dense(1)(h)

    model = Model(inputs=[x], outputs=[out])
    model.compile(optimizer='adam',
                  loss=[get_gaussian_nll(pred_var)],
                  metrics=['mse'])
    return model

def build_vae(latent_dim, n_tokens=4, seq_length=33, enc1_units=50, eps_std=1., ):
    model = SimpleVAE(input_shape=(seq_length, n_tokens,),
                      latent_dim=latent_dim)

    # set encoder layers:
    model.encoderLayers_ = [
        Dense(units=enc1_units, activation='elu', name='e2'),
    ]

    # set decoder layers:
    model.decoderLayers_ = [
        Dense(units=enc1_units, activation='elu', name='d1'),
        Dense(units=n_tokens * seq_length, name='d3'),
        Reshape((seq_length, n_tokens), name='d4'),
        Dense(units=n_tokens, activation='softmax', name='d5'),
    ]

    # build models:
    kl_scale = K.variable(1.)
    model.build_encoder()
    model.build_decoder(decode_activation='softmax')
    model.build_vae(epsilon_std=eps_std, kl_scale=kl_scale)

    losses = [summed_categorical_crossentropy, identity_loss]

    model.compile(optimizer='adam',
                  loss=losses)

    return model

def get_gfp_base_seq():
    lines = open("/global/homes/d/dbrookes/design_icml/data/avGFP_reference_sequence.fa").readlines()
    seq = lines[1].strip()
    return seq
    

def read_gfp_data(path=None, df_save_file=None):
    if path is None:
        path = "/global/homes/d/dbrookes/design_icml/data/nucleotide_genotypes_to_brightness.tsv"
    f = open(path)
    base_seq = get_base_seq()
    mod = list(base_seq)
    mod[191] = 'A'
    base_seq = "".join(mod)
    data = []
    i = 0 
    cols = None
    for line in f:
        if i == 0:
            cols = line.strip().split('\t')
            cols = ['nucSequence', 'numNucMutations', 'numAAMutations'] + cols[2:]
            i += 1
            continue
        else:
            split = line.split('\t')
            mutations = split[0].split(':')
            seq, n_nuc_mut = convert_mutations_to_sequence(base_seq,mutations)
            n_aa_mut = len([m for m in split[1].split(':') if m != ''])
            if split[-1].strip() == '':
                std = 0
            else:
                std = float(split[-1].strip())
            data.append([seq, n_nuc_mut, n_aa_mut, int(split[2]), float(split[3]), std])
            i +=1
    df = pd.DataFrame(data, columns=cols)
    if df_save_file is not None:
        df.to_csv(df_save_file)
    return df

def convert_mutations_to_sequence(base_seq, mutations):
    new_seq = list(base_seq)
    n = 0
    for m in mutations:
        if len(m) == 0:
            continue
        style = m[0]
        assert style == 'S'
        nuc1 = m[1]
        nuc2 = m[-1]
        pos = int(m[2:-1])
        assert new_seq[pos] == nuc1
        new_seq[pos] = nuc2
        n += 1
    new_seq = "".join(new_seq)
    return new_seq, n

def one_hot_encode_dna(dna_str, pad=None, base_order='ATCG'):
    """ Convert length M string into M x 4 tokenized array """
    dna_str = dna_str.upper()
    if pad is not None:
        M = pad
    else:
        M = len(dna_str)
    dna_arr = np.zeros((M, 4))
    for i in range(len(dna_str)):
        idx = base_order.index(dna_str[i])
        dna_arr[i, idx] = 1
    return dna_arr

def one_hot_encode_aa(aa_str, pad=None):
    M = len(aa_str)
    aa_arr = np.zeros((M, 20), dtype=int)
    for i in range(M):
        aa_arr[i, AA_IDX[aa_str[i]]] = 1
    return aa_arr

def convert_aas_to_idx_array(X_aa):
    N = len(X_aa)
    M = len(X_aa[0])
    X_aa_idx = np.zeros((N, M),dtype=int)
    for i in range(N):
        for j in range(M):
            X_aa_idx[i, j] = AA_IDX[X_aa[i][j]]

    return X_aa_idx

def convert_idx_array_to_aas(X_aa):
    N = len(X_aa)
    M = len(X_aa[0])
    X_aa_str = [["A"] * M] * N
    for i in range(N):
        for j in range(M):
            X_aa_str[i][j] = AA[X_aa[i, j]]
        X_aa_str[i] = "".join(X_aa_str[i])
    return X_aa_str


def get_argmax(Xt_p):
    Xt_argmax = np.zeros_like(Xt_p)
    Xt_argmax[np.arange(Xt_p.shape[0]).reshape(Xt_p.shape[0], 1),
                     np.arange(Xt_p.shape[1]).reshape(1, Xt_p.shape[1]),
                     np.argmax(Xt_p, axis=-1)] = 1
    return Xt_argmax

def get_samples(Xt_p):
    Xt_sampled = np.zeros_like(Xt_p)
    for i in range(Xt_p.shape[0]):
        for j in range(Xt_p.shape[1]):
            p = Xt_p[i, j]
            k = np.random.choice(range(len(p)), p=p)
            Xt_sampled[i, j, k] = 1.
    return Xt_sampled

def get_balaji_predictions(preds, Xt):
    M = len(preds)
    N = Xt.shape[0]
    means = np.zeros((M, N))
    variances = np.zeros((M, N))
    for m in range(M):
        y_pred = preds[m].predict(Xt)
#         print(y_pred)
        means[m, :] = y_pred[:, 0]
#         print(y_pred[:, 0].shape, y_pred[:, 1].shape, K.softplus(y_pred[:, 1]).shape, )
        variances[m, :] = np.log(1+np.exp(y_pred[:, 1])) + 1e-6
    mu_star = np.mean(means, axis=0)
    var_star = (1/M) * (np.sum(variances, axis=0) + np.sum(means**2, axis=0)) - mu_star**2
    return mu_star, var_star


def partition_data(X, y, percentile=40, train_size=1000, random_state=1, return_test=False):
    np.random.seed(random_state)
    assert (percentile*0.01 * len(y) >= train_size)
    y_percentile = np.percentile(y, percentile)
    idx = np.where(y < y_percentile)[0]
#     print(y_percentile)
    rand_idx = np.random.choice(idx, size=train_size, replace=False)
    X_train = X[rand_idx]
    y_train = y[rand_idx]
    if return_test:
        test_idx = [i for i in idx if i not in rand_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train

def get_experimental_X_y(random_state=1, train_size=5000, return_test=False, return_all=False):
    """Partition and add noise"""
    df = pd.read_csv('data/gfp_data.csv')
    X,_ = get_gfp_X_y_aa(df, large_only=True, ignore_stops=True)
    y_gt = np.load("data/gfp_gt_evals.npy")
    if return_test:
        X_train, gt_train, X_test, gt_test = partition_data(X, y_gt, percentile=20, train_size=train_size, random_state=random_state, return_test=return_test)
        np.random.seed(random_state)
        gt_var = 0.01
        y_train = gt_train + np.random.randn(*gt_train.shape) * gt_var
        y_test = gt_test + np.random.randn(*gt_test.shape) * gt_var
        return X_train, y_train, gt_train, X_test, y_test, gt_test
    else:
        X_train, gt_train = partition_data(X, y_gt, percentile=20, train_size=train_size, random_state=random_state, return_test=return_test)
        np.random.seed(random_state)
        gt_var = 0.01
        y_train = gt_train + np.random.randn(*gt_train.shape) * gt_var
        return X_train, y_train, gt_train


def get_gfp_X_y_aa(data_df, large_only=False, ignore_stops=True, return_str=False):
    if large_only:
        idx = data_df.loc[(data_df['medianBrightness'] > data_df['medianBrightness'].mean())].index
    else:
        idx = data_df.index
    data_df = data_df.loc[idx]
    
    if ignore_stops:
        idx = data_df.loc[~data_df['aaSequence'].str.contains('!')].index
    data_df = data_df.loc[idx]
    seqs = data_df['aaSequence']
        
    M = len(seqs[0])
    N = len(seqs)
    X = np.zeros((N, M, 20))
    j = 0
    for i in idx:
        X[j] = one_hot_encode_aa(seqs[i])
        j += 1
    y_raw = np.array(data_df['medianBrightness'][idx])
    y = y_raw
    if return_str:
        return X, y, list(data_df['aaSequence'])
    else:
        return X, y