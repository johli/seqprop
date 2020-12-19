import numpy as np
import keras
import tensorflow as tf
import keras.backend as K

#Define kl divergence helper functions (numpy)

def _get_kl_divergence_numpy(p_dist, p_theta, p_phi, p_omega, t_dist, t_theta, t_phi, t_omega) :
    
    kl_dist = np.mean(np.sum(t_dist * np.log(t_dist / p_dist), axis=-1))
    kl_theta = np.mean(np.sum(t_theta * np.log(t_theta / p_theta), axis=-1))
    kl_phi = np.mean(np.sum(t_phi * np.log(t_phi / p_phi), axis=-1))
    kl_omega = np.mean(np.sum(t_omega * np.log(t_omega / p_omega), axis=-1))
    
    return kl_dist, kl_theta, kl_phi, kl_omega

def _get_smooth_kl_divergence_numpy(p_dist, p_theta, p_phi, p_omega, t_dist, t_theta, t_phi, t_omega) :
    
    td0, t1_dist = np.expand_dims(t_dist[..., 0], axis=-1), t_dist[..., 1:]
    tt0, t1_theta = np.expand_dims(t_theta[..., 0], axis=-1), t_theta[..., 1:]
    tp0, t1_phi = np.expand_dims(t_phi[..., 0], axis=-1), t_phi[..., 1:]
    to0, t1_omega = np.expand_dims(t_omega[..., 0], axis=-1), t_omega[..., 1:]

    pd0, p1_dist = np.expand_dims(p_dist[..., 0], axis=-1), p_dist[..., 1:]
    pt0, p1_theta = np.expand_dims(p_theta[..., 0], axis=-1), p_theta[..., 1:]
    pp0, p1_phi = np.expand_dims(p_phi[..., 0], axis=-1), p_phi[..., 1:]
    po0, p1_omega = np.expand_dims(p_omega[..., 0], axis=-1), p_omega[..., 1:]
    
    range_dist = np.linspace(0., 1., t1_dist.shape[3])
    range_theta = np.linspace(0., 1., t1_theta.shape[3])
    range_phi = np.linspace(0., 1., t1_phi.shape[3])
    range_omega = np.linspace(0., 1., t1_omega.shape[3])

    pd1 = np.expand_dims(np.sum(p1_dist * np.tile(np.reshape(range_dist, (1, 1, 1, p1_dist.shape[3])), (1, p1_dist.shape[1], p1_dist.shape[2], 1)), axis=-1), axis=-1)
    pt1 = np.expand_dims(np.sum(p1_theta * np.tile(np.reshape(range_theta, (1, 1, 1, p1_theta.shape[3])), (1, p1_theta.shape[1], p1_theta.shape[2], 1)), axis=-1), axis=-1)
    pp1 = np.expand_dims(np.sum(p1_phi * np.tile(np.reshape(range_phi, (1, 1, 1, p1_phi.shape[3])), (1, p1_phi.shape[1], p1_phi.shape[2], 1)), axis=-1), axis=-1)
    po1 = np.expand_dims(np.sum(p1_omega * np.tile(np.reshape(range_omega, (1, 1, 1, p1_omega.shape[3])), (1, p1_omega.shape[1], p1_omega.shape[2], 1)), axis=-1), axis=-1)

    td1 = np.expand_dims(np.sum(t1_dist * np.tile(np.reshape(range_dist, (1, 1, 1, t1_dist.shape[3])), (1, t1_dist.shape[1], t1_dist.shape[2], 1)), axis=-1), axis=-1)
    tt1 = np.expand_dims(np.sum(t1_theta * np.tile(np.reshape(range_theta, (1, 1, 1, t1_theta.shape[3])), (1, t1_theta.shape[1], t1_theta.shape[2], 1)), axis=-1), axis=-1)
    tp1 = np.expand_dims(np.sum(t1_phi * np.tile(np.reshape(range_phi, (1, 1, 1, t1_phi.shape[3])), (1, t1_phi.shape[1], t1_phi.shape[2], 1)), axis=-1), axis=-1)
    to1 = np.expand_dims(np.sum(t1_omega * np.tile(np.reshape(range_omega, (1, 1, 1, t1_omega.shape[3])), (1, t1_omega.shape[1], t1_omega.shape[2], 1)), axis=-1), axis=-1)

    pd_val = np.clip(np.concatenate([pd0, pd1, 1. - pd0 - pd1], axis=-1), 1e-7, 1. - 1e-7)
    pt_val = np.clip(np.concatenate([pt0, pt1, 1. - pt0 - pt1], axis=-1), 1e-7, 1. - 1e-7)
    pp_val = np.clip(np.concatenate([pp0, pp1, 1. - pp0 - pp1], axis=-1), 1e-7, 1. - 1e-7)
    po_val = np.clip(np.concatenate([po0, po1, 1. - po0 - po1], axis=-1), 1e-7, 1. - 1e-7)

    td_val = np.clip(np.concatenate([td0, td1, 1. - td0 - td1], axis=-1), 1e-7, 1. - 1e-7)
    tt_val = np.clip(np.concatenate([tt0, tt1, 1. - tt0 - tt1], axis=-1), 1e-7, 1. - 1e-7)
    tp_val = np.clip(np.concatenate([tp0, tp1, 1. - tp0 - tp1], axis=-1), 1e-7, 1. - 1e-7)
    to_val = np.clip(np.concatenate([to0, to1, 1. - to0 - to1], axis=-1), 1e-7, 1. - 1e-7)

    kl_dist = np.mean(np.sum(td_val * np.log(td_val / pd_val), axis=-1))
    kl_theta = np.mean(np.sum(tt_val * np.log(tt_val / pt_val), axis=-1))
    kl_phi = np.mean(np.sum(tp_val * np.log(tp_val / pp_val), axis=-1))
    kl_omega = np.mean(np.sum(to_val * np.log(to_val / po_val), axis=-1))
    
    return kl_dist, kl_theta, kl_phi, kl_omega

def _get_smooth_circular_kl_divergence_numpy(p_dist, p_theta, p_phi, p_omega, t_dist, t_theta, t_phi, t_omega) :
    
    td0, t1_dist = np.expand_dims(t_dist[..., 0], axis=-1), t_dist[..., 1:]
    tt0, t1_theta = np.expand_dims(t_theta[..., 0], axis=-1), t_theta[..., 1:]
    tp0, t1_phi = np.expand_dims(t_phi[..., 0], axis=-1), t_phi[..., 1:]
    to0, t1_omega = np.expand_dims(t_omega[..., 0], axis=-1), t_omega[..., 1:]

    pd0, p1_dist = np.expand_dims(p_dist[..., 0], axis=-1), p_dist[..., 1:]
    pt0, p1_theta = np.expand_dims(p_theta[..., 0], axis=-1), p_theta[..., 1:]
    pp0, p1_phi = np.expand_dims(p_phi[..., 0], axis=-1), p_phi[..., 1:]
    po0, p1_omega = np.expand_dims(p_omega[..., 0], axis=-1), p_omega[..., 1:]
    
    range_dist = np.linspace(0., 1., t1_dist.shape[3])
    
    range_sin_theta = (np.sin(np.linspace(-np.pi, np.pi, p1_theta.shape[3])) + 1.) / 2.
    range_cos_theta = (np.cos(np.linspace(-np.pi, np.pi, p1_theta.shape[3])) + 1.) / 2.
    
    range_phi = np.linspace(0., 1., t1_phi.shape[3])
    
    range_sin_omega = (np.sin(np.linspace(-np.pi, np.pi, p1_omega.shape[3])) + 1.) / 2.
    range_cos_omega = (np.cos(np.linspace(-np.pi, np.pi, p1_omega.shape[3])) + 1.) / 2.

    pd1 = np.expand_dims(np.sum(p1_dist * np.tile(np.reshape(range_dist, (1, 1, 1, p1_dist.shape[3])), (1, p1_dist.shape[1], p1_dist.shape[2], 1)), axis=-1), axis=-1)
    pt1_sin = np.expand_dims(np.sum(p1_theta * np.tile(np.reshape(range_sin_theta, (1, 1, 1, p1_theta.shape[3])), (1, p1_theta.shape[1], p1_theta.shape[2], 1)), axis=-1), axis=-1)
    pt1_cos = np.expand_dims(np.sum(p1_theta * np.tile(np.reshape(range_cos_theta, (1, 1, 1, p1_theta.shape[3])), (1, p1_theta.shape[1], p1_theta.shape[2], 1)), axis=-1), axis=-1)
    pp1 = np.expand_dims(np.sum(p1_phi * np.tile(np.reshape(range_phi, (1, 1, 1, p1_phi.shape[3])), (1, p1_phi.shape[1], p1_phi.shape[2], 1)), axis=-1), axis=-1)
    po1_sin = np.expand_dims(np.sum(p1_omega * np.tile(np.reshape(range_sin_omega, (1, 1, 1, p1_omega.shape[3])), (1, p1_omega.shape[1], p1_omega.shape[2], 1)), axis=-1), axis=-1)
    po1_cos = np.expand_dims(np.sum(p1_omega * np.tile(np.reshape(range_cos_omega, (1, 1, 1, p1_omega.shape[3])), (1, p1_omega.shape[1], p1_omega.shape[2], 1)), axis=-1), axis=-1)
    
    td1 = np.expand_dims(np.sum(t1_dist * np.tile(np.reshape(range_dist, (1, 1, 1, t1_dist.shape[3])), (1, t1_dist.shape[1], t1_dist.shape[2], 1)), axis=-1), axis=-1)
    tt1_sin = np.expand_dims(np.sum(t1_theta * np.tile(np.reshape(range_sin_theta, (1, 1, 1, t1_theta.shape[3])), (1, t1_theta.shape[1], t1_theta.shape[2], 1)), axis=-1), axis=-1)
    tt1_cos = np.expand_dims(np.sum(t1_theta * np.tile(np.reshape(range_cos_theta, (1, 1, 1, t1_theta.shape[3])), (1, t1_theta.shape[1], t1_theta.shape[2], 1)), axis=-1), axis=-1)
    tp1 = np.expand_dims(np.sum(t1_phi * np.tile(np.reshape(range_phi, (1, 1, 1, t1_phi.shape[3])), (1, t1_phi.shape[1], t1_phi.shape[2], 1)), axis=-1), axis=-1)
    to1_sin = np.expand_dims(np.sum(t1_omega * np.tile(np.reshape(range_sin_omega, (1, 1, 1, t1_omega.shape[3])), (1, t1_omega.shape[1], t1_omega.shape[2], 1)), axis=-1), axis=-1)
    to1_cos = np.expand_dims(np.sum(t1_omega * np.tile(np.reshape(range_cos_omega, (1, 1, 1, t1_omega.shape[3])), (1, t1_omega.shape[1], t1_omega.shape[2], 1)), axis=-1), axis=-1)
    
    pd_val = np.clip(np.concatenate([pd0, pd1, 1. - pd0 - pd1], axis=-1), 1e-7, 1. - 1e-7)
    pt_sin_val = np.clip(np.concatenate([pt0, pt1_sin, 1. - pt0 - pt1_sin], axis=-1), 1e-7, 1. - 1e-7)
    pt_cos_val = np.clip(np.concatenate([pt0, pt1_cos, 1. - pt0 - pt1_cos], axis=-1), 1e-7, 1. - 1e-7)
    pp_val = np.clip(np.concatenate([pp0, pp1, 1. - pp0 - pp1], axis=-1), 1e-7, 1. - 1e-7)
    po_sin_val = np.clip(np.concatenate([po0, po1_sin, 1. - po0 - po1_sin], axis=-1), 1e-7, 1. - 1e-7)
    po_cos_val = np.clip(np.concatenate([po0, po1_cos, 1. - po0 - po1_cos], axis=-1), 1e-7, 1. - 1e-7)

    td_val = np.clip(np.concatenate([td0, td1, 1. - td0 - td1], axis=-1), 1e-7, 1. - 1e-7)
    tt_sin_val = np.clip(np.concatenate([tt0, tt1_sin, 1. - tt0 - tt1_sin], axis=-1), 1e-7, 1. - 1e-7)
    tt_cos_val = np.clip(np.concatenate([tt0, tt1_cos, 1. - tt0 - tt1_cos], axis=-1), 1e-7, 1. - 1e-7)
    tp_val = np.clip(np.concatenate([tp0, tp1, 1. - tp0 - tp1], axis=-1), 1e-7, 1. - 1e-7)
    to_sin_val = np.clip(np.concatenate([to0, to1_sin, 1. - to0 - to1_sin], axis=-1), 1e-7, 1. - 1e-7)
    to_cos_val = np.clip(np.concatenate([to0, to1_cos, 1. - to0 - to1_cos], axis=-1), 1e-7, 1. - 1e-7)

    kl_dist = np.mean(np.sum(td_val * np.log(td_val / pd_val), axis=-1))
    kl_theta_sin = np.mean(np.sum(tt_sin_val * np.log(tt_sin_val / pt_sin_val), axis=-1)) * 0.5
    kl_theta_cos = np.mean(np.sum(tt_cos_val * np.log(tt_cos_val / pt_cos_val), axis=-1)) * 0.5
    kl_phi = np.mean(np.sum(tp_val * np.log(tp_val / pp_val), axis=-1))
    kl_omega_sin = np.mean(np.sum(to_sin_val * np.log(to_sin_val / po_sin_val), axis=-1)) * 0.5
    kl_omega_cos = np.mean(np.sum(to_cos_val * np.log(to_cos_val / po_cos_val), axis=-1)) * 0.5
    
    return kl_dist, kl_theta_sin, kl_theta_cos, kl_phi, kl_omega_sin, kl_omega_cos

#Define kl divergence helper functions (keras)

def _get_kl_divergence_keras(p_dist, p_theta, p_phi, p_omega, target_p_dist, target_p_theta, target_p_phi, target_p_omega) :
    
    t_dist = K.clip(K.constant(target_p_dist), K.epsilon(), 1. - K.epsilon())
    t_theta = K.clip(K.constant(target_p_theta), K.epsilon(), 1. - K.epsilon())
    t_phi = K.clip(K.constant(target_p_phi), K.epsilon(), 1. - K.epsilon())
    t_omega = K.clip(K.constant(target_p_omega), K.epsilon(), 1. - K.epsilon())
    
    kl_dist = K.mean(K.sum(t_dist * K.log(t_dist / p_dist), axis=-1), axis=(-1, -2))
    kl_theta = K.mean(K.sum(t_theta * K.log(t_theta / p_theta), axis=-1), axis=(-1, -2))
    kl_phi = K.mean(K.sum(t_phi * K.log(t_phi / p_phi), axis=-1), axis=(-1, -2))
    kl_omega = K.mean(K.sum(t_omega * K.log(t_omega / p_omega), axis=-1), axis=(-1, -2))
    
    return kl_dist, kl_theta, kl_phi, kl_omega

def _get_smooth_kl_divergence_keras(p_dist, p_theta, p_phi, p_omega, target_p_dist, target_p_theta, target_p_phi, target_p_omega) :
    
    t_dist = K.clip(K.constant(target_p_dist), K.epsilon(), 1. - K.epsilon())
    t_theta = K.clip(K.constant(target_p_theta), K.epsilon(), 1. - K.epsilon())
    t_phi = K.clip(K.constant(target_p_phi), K.epsilon(), 1. - K.epsilon())
    t_omega = K.clip(K.constant(target_p_omega), K.epsilon(), 1. - K.epsilon())
    
    td0, t1_dist = K.expand_dims(t_dist[..., 0], axis=-1), t_dist[..., 1:]
    tt0, t1_theta = K.expand_dims(t_theta[..., 0], axis=-1), t_theta[..., 1:]
    tp0, t1_phi = K.expand_dims(t_phi[..., 0], axis=-1), t_phi[..., 1:]
    to0, t1_omega = K.expand_dims(t_omega[..., 0], axis=-1), t_omega[..., 1:]

    pd0, p1_dist = K.expand_dims(p_dist[..., 0], axis=-1), p_dist[..., 1:]
    pt0, p1_theta = K.expand_dims(p_theta[..., 0], axis=-1), p_theta[..., 1:]
    pp0, p1_phi = K.expand_dims(p_phi[..., 0], axis=-1), p_phi[..., 1:]
    po0, p1_omega = K.expand_dims(p_omega[..., 0], axis=-1), p_omega[..., 1:]
    
    range_dist = np.linspace(0., 1., target_p_dist.shape[3] - 1)
    range_theta = np.linspace(0., 1., target_p_theta.shape[3] - 1)
    range_phi = np.linspace(0., 1., target_p_phi.shape[3] - 1)
    range_omega = np.linspace(0., 1., target_p_omega.shape[3] - 1)

    pd1 = K.expand_dims(K.sum(p1_dist * K.tile(K.reshape(K.constant(range_dist), (1, 1, 1, target_p_dist.shape[3] - 1)), (1, target_p_dist.shape[1], target_p_dist.shape[2], 1)), axis=-1), axis=-1)
    pt1 = K.expand_dims(K.sum(p1_theta * K.tile(K.reshape(K.constant(range_theta), (1, 1, 1, target_p_theta.shape[3] - 1)), (1, target_p_theta.shape[1], target_p_theta.shape[2], 1)), axis=-1), axis=-1)
    pp1 = K.expand_dims(K.sum(p1_phi * K.tile(K.reshape(K.constant(range_phi), (1, 1, 1, target_p_phi.shape[3] - 1)), (1, target_p_phi.shape[1], target_p_phi.shape[2], 1)), axis=-1), axis=-1)
    po1 = K.expand_dims(K.sum(p1_omega * K.tile(K.reshape(K.constant(range_omega), (1, 1, 1, target_p_omega.shape[3] - 1)), (1, target_p_omega.shape[1], target_p_omega.shape[2], 1)), axis=-1), axis=-1)

    td1 = K.expand_dims(K.sum(t1_dist * K.tile(K.reshape(K.constant(range_dist), (1, 1, 1, target_p_dist.shape[3] - 1)), (1, target_p_dist.shape[1], target_p_dist.shape[2], 1)), axis=-1), axis=-1)
    tt1 = K.expand_dims(K.sum(t1_theta * K.tile(K.reshape(K.constant(range_theta), (1, 1, 1, target_p_theta.shape[3] - 1)), (1, target_p_theta.shape[1], target_p_theta.shape[2], 1)), axis=-1), axis=-1)
    tp1 = K.expand_dims(K.sum(t1_phi * K.tile(K.reshape(K.constant(range_phi), (1, 1, 1, target_p_phi.shape[3] - 1)), (1, target_p_phi.shape[1], target_p_phi.shape[2], 1)), axis=-1), axis=-1)
    to1 = K.expand_dims(K.sum(t1_omega * K.tile(K.reshape(K.constant(range_omega), (1, 1, 1, target_p_omega.shape[3] - 1)), (1, target_p_omega.shape[1], target_p_omega.shape[2], 1)), axis=-1), axis=-1)

    pd_val = K.clip(K.concatenate([pd0, pd1, 1. - pd0 - pd1], axis=-1), K.epsilon(), 1. - K.epsilon())
    pt_val = K.clip(K.concatenate([pt0, pt1, 1. - pt0 - pt1], axis=-1), K.epsilon(), 1. - K.epsilon())
    pp_val = K.clip(K.concatenate([pp0, pp1, 1. - pp0 - pp1], axis=-1), K.epsilon(), 1. - K.epsilon())
    po_val = K.clip(K.concatenate([po0, po1, 1. - po0 - po1], axis=-1), K.epsilon(), 1. - K.epsilon())

    td_val = K.clip(K.concatenate([td0, td1, 1. - td0 - td1], axis=-1), K.epsilon(), 1. - K.epsilon())
    tt_val = K.clip(K.concatenate([tt0, tt1, 1. - tt0 - tt1], axis=-1), K.epsilon(), 1. - K.epsilon())
    tp_val = K.clip(K.concatenate([tp0, tp1, 1. - tp0 - tp1], axis=-1), K.epsilon(), 1. - K.epsilon())
    to_val = K.clip(K.concatenate([to0, to1, 1. - to0 - to1], axis=-1), K.epsilon(), 1. - K.epsilon())

    kl_dist = K.mean(K.sum(td_val * K.log(td_val / pd_val), axis=-1), axis=(-1, -2))
    kl_theta = K.mean(K.sum(tt_val * K.log(tt_val / pt_val), axis=-1), axis=(-1, -2))
    kl_phi = K.mean(K.sum(tp_val * K.log(tp_val / pp_val), axis=-1), axis=(-1, -2))
    kl_omega = K.mean(K.sum(to_val * K.log(to_val / po_val), axis=-1), axis=(-1, -2))
    
    return kl_dist, kl_theta, kl_phi, kl_omega

def _get_smooth_circular_kl_divergence_keras(p_dist, p_theta, p_phi, p_omega, target_p_dist, target_p_theta, target_p_phi, target_p_omega) :
    
    t_dist = K.clip(K.constant(target_p_dist), K.epsilon(), 1. - K.epsilon())
    t_theta = K.clip(K.constant(target_p_theta), K.epsilon(), 1. - K.epsilon())
    t_phi = K.clip(K.constant(target_p_phi), K.epsilon(), 1. - K.epsilon())
    t_omega = K.clip(K.constant(target_p_omega), K.epsilon(), 1. - K.epsilon())
    
    td0, t1_dist = K.expand_dims(t_dist[..., 0], axis=-1), t_dist[..., 1:]
    tt0, t1_theta = K.expand_dims(t_theta[..., 0], axis=-1), t_theta[..., 1:]
    tp0, t1_phi = K.expand_dims(t_phi[..., 0], axis=-1), t_phi[..., 1:]
    to0, t1_omega = K.expand_dims(t_omega[..., 0], axis=-1), t_omega[..., 1:]

    pd0, p1_dist = K.expand_dims(p_dist[..., 0], axis=-1), p_dist[..., 1:]
    pt0, p1_theta = K.expand_dims(p_theta[..., 0], axis=-1), p_theta[..., 1:]
    pp0, p1_phi = K.expand_dims(p_phi[..., 0], axis=-1), p_phi[..., 1:]
    po0, p1_omega = K.expand_dims(p_omega[..., 0], axis=-1), p_omega[..., 1:]
    
    range_dist = np.linspace(0., 1., target_p_dist.shape[3] - 1)
    
    range_sin_theta = (np.sin(np.linspace(-np.pi, np.pi, target_p_theta.shape[3] - 1)) + 1.) / 2.
    range_cos_theta = (np.cos(np.linspace(-np.pi, np.pi, target_p_theta.shape[3] - 1)) + 1.) / 2.
    
    range_phi = np.linspace(0., 1., target_p_phi.shape[3] - 1)
    
    range_sin_omega = (np.sin(np.linspace(-np.pi, np.pi, target_p_omega.shape[3] - 1)) + 1.) / 2.
    range_cos_omega = (np.cos(np.linspace(-np.pi, np.pi, target_p_omega.shape[3] - 1)) + 1.) / 2.

    pd1 = K.expand_dims(K.sum(p1_dist * K.tile(K.reshape(K.constant(range_dist), (1, 1, 1, target_p_dist.shape[3] - 1)), (1, target_p_dist.shape[1], target_p_dist.shape[2], 1)), axis=-1), axis=-1)
    pt1_sin = K.expand_dims(K.sum(p1_theta * K.tile(K.reshape(K.constant(range_sin_theta), (1, 1, 1, target_p_theta.shape[3] - 1)), (1, target_p_theta.shape[1], target_p_theta.shape[2], 1)), axis=-1), axis=-1)
    pt1_cos = K.expand_dims(K.sum(p1_theta * K.tile(K.reshape(K.constant(range_cos_theta), (1, 1, 1, target_p_theta.shape[3] - 1)), (1, target_p_theta.shape[1], target_p_theta.shape[2], 1)), axis=-1), axis=-1)
    pp1 = K.expand_dims(K.sum(p1_phi * K.tile(K.reshape(K.constant(range_phi), (1, 1, 1, target_p_phi.shape[3] - 1)), (1, target_p_phi.shape[1], target_p_phi.shape[2], 1)), axis=-1), axis=-1)
    po1_sin = K.expand_dims(K.sum(p1_omega * K.tile(K.reshape(K.constant(range_sin_omega), (1, 1, 1, target_p_omega.shape[3] - 1)), (1, target_p_omega.shape[1], target_p_omega.shape[2], 1)), axis=-1), axis=-1)
    po1_cos = K.expand_dims(K.sum(p1_omega * K.tile(K.reshape(K.constant(range_cos_omega), (1, 1, 1, target_p_omega.shape[3] - 1)), (1, target_p_omega.shape[1], target_p_omega.shape[2], 1)), axis=-1), axis=-1)

    td1 = K.expand_dims(K.sum(t1_dist * K.tile(K.reshape(K.constant(range_dist), (1, 1, 1, target_p_dist.shape[3] - 1)), (1, target_p_dist.shape[1], target_p_dist.shape[2], 1)), axis=-1), axis=-1)
    tt1_sin = K.expand_dims(K.sum(t1_theta * K.tile(K.reshape(K.constant(range_sin_theta), (1, 1, 1, target_p_theta.shape[3] - 1)), (1, target_p_theta.shape[1], target_p_theta.shape[2], 1)), axis=-1), axis=-1)
    tt1_cos = K.expand_dims(K.sum(t1_theta * K.tile(K.reshape(K.constant(range_cos_theta), (1, 1, 1, target_p_theta.shape[3] - 1)), (1, target_p_theta.shape[1], target_p_theta.shape[2], 1)), axis=-1), axis=-1)
    tp1 = K.expand_dims(K.sum(t1_phi * K.tile(K.reshape(K.constant(range_phi), (1, 1, 1, target_p_phi.shape[3] - 1)), (1, target_p_phi.shape[1], target_p_phi.shape[2], 1)), axis=-1), axis=-1)
    to1_sin = K.expand_dims(K.sum(t1_omega * K.tile(K.reshape(K.constant(range_sin_omega), (1, 1, 1, target_p_omega.shape[3] - 1)), (1, target_p_omega.shape[1], target_p_omega.shape[2], 1)), axis=-1), axis=-1)
    to1_cos = K.expand_dims(K.sum(t1_omega * K.tile(K.reshape(K.constant(range_cos_omega), (1, 1, 1, target_p_omega.shape[3] - 1)), (1, target_p_omega.shape[1], target_p_omega.shape[2], 1)), axis=-1), axis=-1)
    
    pd_val = K.clip(K.concatenate([pd0, pd1, 1. - pd0 - pd1], axis=-1), K.epsilon(), 1. - K.epsilon())
    pt_sin_val = K.clip(K.concatenate([pt0, pt1_sin, 1. - pt0 - pt1_sin], axis=-1), K.epsilon(), 1. - K.epsilon())
    pt_cos_val = K.clip(K.concatenate([pt0, pt1_cos, 1. - pt0 - pt1_cos], axis=-1), K.epsilon(), 1. - K.epsilon())
    pp_val = K.clip(K.concatenate([pp0, pp1, 1. - pp0 - pp1], axis=-1), K.epsilon(), 1. - K.epsilon())
    po_sin_val = K.clip(K.concatenate([po0, po1_sin, 1. - po0 - po1_sin], axis=-1), K.epsilon(), 1. - K.epsilon())
    po_cos_val = K.clip(K.concatenate([po0, po1_cos, 1. - po0 - po1_cos], axis=-1), K.epsilon(), 1. - K.epsilon())

    td_val = K.clip(K.concatenate([td0, td1, 1. - td0 - td1], axis=-1), K.epsilon(), 1. - K.epsilon())
    tt_sin_val = K.clip(K.concatenate([tt0, tt1_sin, 1. - tt0 - tt1_sin], axis=-1), K.epsilon(), 1. - K.epsilon())
    tt_cos_val = K.clip(K.concatenate([tt0, tt1_cos, 1. - tt0 - tt1_cos], axis=-1), K.epsilon(), 1. - K.epsilon())
    tp_val = K.clip(K.concatenate([tp0, tp1, 1. - tp0 - tp1], axis=-1), K.epsilon(), 1. - K.epsilon())
    to_sin_val = K.clip(K.concatenate([to0, to1_sin, 1. - to0 - to1_sin], axis=-1), K.epsilon(), 1. - K.epsilon())
    to_cos_val = K.clip(K.concatenate([to0, to1_cos, 1. - to0 - to1_cos], axis=-1), K.epsilon(), 1. - K.epsilon())

    kl_dist = K.mean(K.sum(td_val * K.log(td_val / pd_val), axis=-1), axis=(-1, -2))
    kl_theta_sin = K.mean(K.sum(tt_sin_val * K.log(tt_sin_val / pt_sin_val), axis=-1), axis=(-1, -2)) * 0.5
    kl_theta_cos = K.mean(K.sum(tt_cos_val * K.log(tt_cos_val / pt_cos_val), axis=-1), axis=(-1, -2)) * 0.5
    kl_phi = K.mean(K.sum(tp_val * K.log(tp_val / pp_val), axis=-1), axis=(-1, -2))
    kl_omega_sin = K.mean(K.sum(to_sin_val * K.log(to_sin_val / po_sin_val), axis=-1), axis=(-1, -2)) * 0.5
    kl_omega_cos = K.mean(K.sum(to_cos_val * K.log(to_cos_val / po_cos_val), axis=-1), axis=(-1, -2)) * 0.5
    
    return kl_dist, kl_theta_sin, kl_theta_cos, kl_phi, kl_omega_sin, kl_omega_cos
