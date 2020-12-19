import numpy as np
from keras import backend as K

from seqtools import SequenceTools


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

def identity_loss(y_true, y_pred):
    return y_pred

def summed_binary_crossentropy(y_true, y_pred):
    """ Negative log likehood of binomial distribution """
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)  # default is mean over last axis


def summed_categorical_crossentropy(y_true, y_pred):
    """ Negative log likelihood of categorical distribution """
    return K.sum(K.categorical_crossentropy(y_true, y_pred), axis=-1)


def get_gaussian_nll(variance=1.):
    """  Returns gaussian negative log likelihood loss function """

    def gaussian_nll(y_true, y_pred):
        return K.sum(0.5 * K.log(2 * np.pi) + 0.5 * K.log(variance) + (0.5 / variance) * K.square(y_true - y_pred),
                     axis=-1)

    return gaussian_nll

def neg_log_likelihood(y_true, y_pred):
    y_true = y_true[:, 0]
    mean = y_pred[:, 0]
    variance = K.softplus(y_pred[:, 1]) + 1e-6
    log_variance = K.log(variance)
    return 0.5 * K.mean(log_variance, axis = -1) + 0.5 * K.mean(K.square(y_true - mean) / variance, axis = -1) + 0.5 * K.log(2 * np.pi)


def get_uncertainty_loss(variance=1.):
    def uncertainty_loss(v_true, v_pred):
        return K.sum((0.5 / variance) * K.square(v_true - v_pred), axis=-1)

    return uncertainty_loss


def get_gaussian_nll_for_log_pred(variance=1.):
    """  Returns gaussian negative log likelihood loss function """

    def gaussian_log_nll(y_true, y_pred):
        return K.sum(
            0.5 * K.log(2 * np.pi) + 0.5 * K.log(variance) + (0.5 / variance) * K.square(y_true - K.exp(y_pred)),
            axis=-1)

    return gaussian_log_nll


def seq_reconstruction_with_translation_loss(y_true, y_pred):
    """ TODO: This is not quite right!! probability should be zero if codon does not map to amino acid"""
    N = y_pred.shape[1].value
    aa_codons = K.constant(SequenceTools.get_aa_codons())

    y_true_bool = K.tf.cast(y_true, K.tf.bool)
    masked = K.tf.where(y_true_bool,
                        y_pred,
                        K.tf.zeros_like(y_pred))
    loss1 = -K.log(K.sum(masked, axis=-1))
    loss1 = K.sum(loss1, axis=-1)

    loss2 = K.tf.zeros_like(loss1)
    y_pred_split = K.tf.stack(K.tf.split(y_pred, int(N / 3), axis=1), axis=1)
    y_true_split = K.tf.stack(K.tf.split(y_true, int(N / 3), axis=1), axis=1)

    y_pred_expand = K.tf.expand_dims(y_pred_split, 2)
    y_pred_tile = K.tf.tile(y_pred_expand, [1, 1, 6, 1, 1])

    y_true_expand = K.tf.expand_dims(y_true_split, 2)
    y_true_expand = K.tf.expand_dims(y_true_expand, 2)
    aa_codons_expand = K.tf.expand_dims(aa_codons, 0)
    aa_codons_expand = K.tf.expand_dims(aa_codons_expand, 0)

    mult = y_true_expand * aa_codons_expand
    aa_idx = K.argmax(K.max(K.sum(mult, axis=(-2, -1)), axis=3), axis=2)
    cods = K.tf.gather(aa_codons, aa_idx)
    cods_bool = K.tf.cast(cods, K.tf.bool)

    mask2 = K.tf.where(cods_bool, y_pred_tile, K.tf.zeros_like(y_pred_tile))
    inner_loss = K.sum(mask2, axis=-1)
    inner_loss = K.prod(inner_loss, axis=-1)
    inner_loss = -K.log(K.sum(inner_loss, axis=-1))
    inner_loss = K.sum(inner_loss, axis=-1)
    loss2 += inner_loss

    losses = loss1 + loss2
    return losses

