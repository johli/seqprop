import keras
import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Layer, Input, Lambda, Add, Multiply, Dense, Flatten, Concatenate, Reshape, Conv1D
from keras.models import Model
from keras import layers

"""
Module for extendable variational autoencoders.

Some code adapted from Louis Tiao's blog: http://louistiao.me/
"""


class KLDivergenceLayer(Layer):
    """ Add KL divergence in latent layer to loss """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, scale=1.):
        """ Add KL loss, then return inputs """

        mu, log_var = inputs
        inner = 1 + log_var - K.square(mu) - K.exp(log_var)

        # sum over dimensions of latent space
        kl_batch = -0.5 * K.sum(inner, axis=1)

        # add mean KL loss over batch
        self.add_loss(scale * K.mean(kl_batch, axis=0), inputs=inputs)
        return mu, log_var


class KLScaleUpdate(Callback):
    """ Callback for updating the scale of the the KL divergence loss

    See Bowman et. al (2016) for motivation on adjusting the scale of the
    KL loss. This class implements a sigmoidal growth, as in Bowman, et. al.

    """

    def __init__(self, scale, growth=0.01, start=0.001, verbose=True):
        super(KLScaleUpdate, self).__init__()
        self.scale_ = scale
        self.start_ = start
        self.growth_ = growth
        self.step_ = 0
        self.verbose_ = verbose

    def on_batch_begin(self, batch, logs=None):
        K.set_value(self.scale_, self._get_next_val(self.step_))
        self.step_ += 1

    def _get_next_val(self, step):
        return 1 - (1 / (1 + self.start_ * np.exp(step * self.growth_)))

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose_ > 0:
            print("KL Divergence weight: %.3f" % K.get_value(self.scale_))


class BaseVAE(object):
    """ Base class for Variational Autoencoders implemented in Keras

    The class is designed to connect user-specified encoder and decoder
    models via a Model representing the latent space

    """

    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        self.latentDim_ = latent_dim
        self.inputShape_ = input_shape

        self.encoder_ = None
        self.decoder_ = None

        self.vae_ = None

    def build_encoder(self, *args, **kwargs):
        """ Build the encoder network as a keras Model

        The encoder Model must ouput the mean and log variance of
        the latent space embeddings. I.e. this model must output
        mu and Sigma of the latent space distribution:

                    q(z|x) = N(z| mu(x), Sigma(x))

        Sets the value of self.encoder_ to a keras Model

        """

        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        """ Build the decoder network as a keras Model

        The input to the decoder must have the same shape as the latent
        space and the output must have the same shape as the input to
        the encoder.

        Sets the value of self.decoder_ to a keras Model

        """

        raise NotImplementedError

    def _build_latent_vars(self, mu_z, log_var_z, epsilon_std=1., kl_scale=1.):
        """ Build keras variables representing the latent space

        First, calculate the KL divergence from the input mean and log variance
        and add this to the model loss via a KLDivergenceLayer. Then sample an epsilon
        and perform a location-scale transformation to obtain the latent embedding, z.

        Args:
            epsilon_std: standard deviation of p(epsilon)
            kl_scale: weight of KL divergence loss

        Returns:
            Variables representing z and epsilon

        """

        # mu_z, log_var_z, kl_batch  = KLDivergenceLayer()([mu_z, log_var_z], scale=kl_scale)
        lmda_func = lambda inputs: -0.5 * K.sum(1 + inputs[1] - K.square(inputs[0]) - K.exp(inputs[1]), axis=1)

        kl_batch = Lambda(lmda_func, name='kl_calc')([mu_z, log_var_z])
        kl_batch = Reshape((1,), name='kl_reshape')(kl_batch)

        # get standard deviation from log variance:
        sigma_z = Lambda(lambda lv: K.exp(0.5 * lv))(log_var_z)

        # re-parametrization trick ( z = mu_z + eps * sigma_z)
        eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                           shape=(K.shape(mu_z)[0], self.latentDim_)))

        eps_z = Multiply()([sigma_z, eps])  # scale by epsilon sample
        z = Add()([mu_z, eps_z])

        return z, eps, kl_batch

    def _get_decoder_input(self, z, enc_in):
        return z

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        """ Build the VAE

        Sets the value of self.vae_ to a keras model

        Args:
            epsilon_std (float): standard deviation of distribution used to sample z via the reparameratization trick
            kl_scale (float or keras.backend.Variable): weight of KL divergence loss

        """

        if self.encoder_ is None:
            raise TypeError("Encoder must be built before calling build_vae")
        if self.decoder_ is None:
            raise TypeError("Decoder must be built before calling build_vae")

        enc_in = self.encoder_.inputs
        mu_z, log_var_z = self.encoder_.outputs
        z, eps, kl_batch = self._build_latent_vars(mu_z, log_var_z, epsilon_std=epsilon_std, kl_scale=kl_scale)
        dec_in = self._get_decoder_input(z, enc_in)
        x_pred = self.decoder_(dec_in)
        self.vae_ = Model(inputs=enc_in + [eps], outputs=[x_pred, kl_batch], name='vae_base')

    def plot_model(self, *args, **kwargs):
        keras.utils.plot_model(self.vae_, *args, **kwargs)

    def compile(self, *args, **kwargs):
        self.vae_.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.vae_.fit(*args, **kwargs)

    def save_all_weights(self, prefix):
        encoder_file = prefix + "_encoder.h5"
        decoder_file = prefix + "_decoder.h5"
        vae_file = prefix + "_vae.h5"

        self.encoder_.save_weights(encoder_file)
        self.decoder_.save_weights(decoder_file)
        self.vae_.save_weights(vae_file)

    def load_all_weights(self, prefix):
        encoder_file = prefix + "_encoder.h5"
        decoder_file = prefix + "_decoder.h5"
        vae_file = prefix + "_vae.h5"

        self.encoder_.load_weights(encoder_file)
        self.decoder_.load_weights(decoder_file)
        self.vae_.load_weights(vae_file)


class BaseConditionalVAE(BaseVAE):
    """ Base class for conditional VAEs (cVAEs) """

    def __init__(self, input_shape, latent_dim, cond_shape=None, *args, **kwargs):
        super(BaseConditionalVAE, self).__init__(input_shape=input_shape,
                                                 latent_dim=latent_dim,
                                                 *args, **kwargs)

        self.condShape_ = cond_shape

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        raise NotImplementedError

    def _get_decoder_input(self, z, enc_in):
        c = enc_in[1]
        dec_in = [z, c]
        return dec_in


class BaseSupervisedVAE(BaseVAE):
    """ Base class for VAEs that also make predictions from the latent space """

    def __init__(self, input_shape, latent_dim, pred_dim,
                 learn_uncertainty=False, pred_var=0.1, *args, **kwargs):
        super(BaseSupervisedVAE, self).__init__(input_shape=input_shape,
                                                latent_dim=latent_dim,
                                                *args, **kwargs)
        self.predDim_ = pred_dim
        self.predictor_ = None
        self.learnUncertainty_ = learn_uncertainty
        self.predVar_ = pred_var

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_predictor(self, *args, **kwargs):
        """ Build the predictor network as a keras Model

        The input to the predictor must have the same shape as the latent
        space and the output must have self.predShape_

        Sets the value of self.predictor_ to a keras Model

        """
        raise NotImplementedError

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        """ Build the VAE

        Sets the value of self.vae_ to a keras model

        Args:
            epsilon_std (float): standard deviation of distribution used to sample z via the reparameratization trick
            kl_scale (float or keras.backend.Variable): weight of KL divergence loss

        """

        if self.encoder_ is None:
            raise TypeError("Encoder must be built before calling build_vae")
        if self.decoder_ is None:
            raise TypeError("Decoder must be built before calling build_vae")

        enc_in = self.encoder_.inputs
        mu_z, log_var_z = self.encoder_.outputs
        z, eps, kl_batch = self._build_latent_vars(mu_z, log_var_z, epsilon_std=epsilon_std, kl_scale=kl_scale)
        dec_in = self._get_decoder_input(z, enc_in)
        x_pred = self.decoder_(dec_in)
        y_pred, y_log_var = self.predictor_(dec_in)
        self.vae_ = Model(inputs=enc_in + [eps], outputs=[x_pred, kl_batch, y_pred, y_log_var], name='vae_pred')

    def save_all_weights(self, prefix):
        super(BaseSupervisedVAE, self).save_all_weights(prefix)
        predictor_file = prefix + "_predictor.h5"
        self.predictor_.save_weights(predictor_file)

    def load_all_weights(self, prefix):
        super(BaseSupervisedVAE, self).load_all_weights(prefix)
        predictor_file = prefix + "_predictor.h5"
        self.predictor_.load_weights(predictor_file)


class BaseSemiSupervisedVAE(BaseSupervisedVAE):
    """ Base class for VAE's with semi-supervised learning """

    def __init__(self, input_shape, latent_dim, pred_dim,
                 learn_uncertainty=False, pred_var=0.1, *args, **kwargs):
        super(BaseSemiSupervisedVAE, self).__init__(input_shape=input_shape,
                                                    latent_dim=latent_dim,
                                                    pred_dim=pred_dim,
                                                    learn_uncertainty=learn_uncertainty,
                                                    pred_var=pred_var,
                                                    *args, **kwargs)

        self.uvae_ = None  # unsupervised model

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_predictor(self, *args, **kwargs):
        raise NotImplementedError

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        """ Build the VAE

        Sets the value of self.vae_ to a keras model

        Args:
            epsilon_std (float): standard deviation of distribution used to sample z via the reparameratization trick
            kl_scale (float or keras.backend.Variable): weight of KL divergence loss

        """

        if self.encoder_ is None:
            raise TypeError("Encoder must be built before calling build_vae")
        if self.decoder_ is None:
            raise TypeError("Decoder must be built before calling build_vae")

        enc_in = self.encoder_.inputs
        mu_z, log_var_z = self.encoder_.outputs
        z, eps = self._build_latent_vars(mu_z, log_var_z, epsilon_std=epsilon_std, kl_scale=kl_scale)
        dec_in = self._get_decoder_input(z, enc_in)
        x_pred = self.decoder_(dec_in)
        y_pred, y_log_var = self.predictor_(dec_in)
        self.vae_ = Model(inputs=enc_in + [eps], outputs=[x_pred, y_pred, y_log_var], name='vae_pred')
        self.uvae_ = Model(inputs=enc_in + [eps], outputs=[x_pred], name='uvae')


class SimpleVAE(BaseVAE):
    """ Basic VAE where the encoder and decoder can be constructed from lists of layers """

    def __init__(self, input_shape, latent_dim, flatten=True, *args, **kwargs):
        super(SimpleVAE, self).__init__(input_shape=input_shape,
                                        latent_dim=latent_dim,
                                        *args, **kwargs)
        self.flatten_ = flatten
        self.encoderLayers_ = []
        self.decoderLayers_ = []

    def add_encoder_layer(self, layer):
        """ Append a keras Layer to self.encoderLayers_"""
        self.encoderLayers_.append(layer)

    def add_decoder_layer(self, layer):
        """ Append a keras Layer to self.decoderLayers_ """
        self.decoderLayers_.append(layer)

    def _build_encoder_inputs(self):
        """ BUILD (as opposed to get) the encoder inputs """
        x = Input(shape=self.inputShape_)
        return [x]

    def _build_decoder_inputs(self):
        z = Input(shape=(self.latentDim_,))
        return z

    def _edit_encoder_inputs(self, enc_in):
        if self.flatten_:
            h = Flatten()(enc_in[0])
        else:
            h = enc_in[0]
        return h

    def _edit_decoder_inputs(self, dec_in):
        return dec_in

    def build_encoder(self):
        """ Construct the encoder from list of layers

        After the final layer in self.encoderLayers_, two Dense layers
        are applied to output mu_z and log_var_z

        """

        if len(self.encoderLayers_) == 0:
            raise ValueError("Must add at least one encoder hidden layer")

        enc_in = self._build_encoder_inputs()
        h = self._edit_encoder_inputs(enc_in)
        for hid in self.encoderLayers_:
            h = hid(h)

        mu_z = Dense(self.latentDim_, name='mu_z')(h)
        log_var_z = Dense(self.latentDim_, name='log_var_z')(h)

        self.encoder_ = Model(inputs=enc_in, outputs=[mu_z, log_var_z], name='encoder')

    def build_decoder(self, decode_activation):
        """ Construct the decoder from list of layers

        After the final layer in self.decoderLayers_, a Dense layer is
        applied to output the final reconstruction

        Args:
            decode_activation: activation of the final decoding layer

        """

        if len(self.decoderLayers_) == 0:
            raise ValueError("Must add at least one decoder hidden layer")

        dec_in = self._build_decoder_inputs()
        h = self._edit_decoder_inputs(dec_in)
        for hid in self.decoderLayers_:
            h = hid(h)

        x_pred = h
        self.decoder_ = Model(inputs=dec_in, outputs=x_pred, name='decoder')
        
        
class BigVAE(BaseVAE):
    """ VAE based on WGAN architecture """

    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        super(BigVAE, self).__init__(input_shape=input_shape,
                                        latent_dim=latent_dim,
                                        *args, **kwargs)

    def build_encoder(self):
        """ Construct the encoder from list of layers

        After the final layer in self.encoderLayers_, two Dense layers
        are applied to output mu_z and log_var_z

        """
        L = self.inputShape_[0]
        x = Input(shape=self.inputShape_)
        y = Conv1D(100, 1, padding='same')(x)

        # res block 1:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])

        # res block 2:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])

        # res block 3:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])

        # res block 4:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])

        # res block 5:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])
        y = Reshape((L * 100,))(y)
        # for hid in self.encoderLayers_:
        #     h = hid(h)

        mu_z = Dense(self.latentDim_, name='mu_z')(y)
        log_var_z = Dense(self.latentDim_, name='log_var_z')(y)

        self.encoder_ = Model(inputs=x, outputs=[mu_z, log_var_z], name='encoder')

    def build_decoder(self):
        """ Construct the decoder from list of layers

        After the final layer in self.decoderLayers_, a Dense layer is
        applied to output the final reconstruction

        Args:
            decode_activation: activation of the final decoding layer

        """
        L = self.inputShape_[0]
        z = Input(shape=(self.latentDim_,))
        x = Dense(100 * self.inputShape_[0])(z)
        x = Reshape((L, 100,))(x)

        # res block 1:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        # res block 2:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        # res block 3:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        # res block 4:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        # res block 5:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        x = Conv1D(4, 1, padding='same')(x)
        out = layers.Activation('softmax')(x)
        self.decoder_ = Model(inputs=z, outputs=out, name='decoder')


class SimpleSupervisedVAE(SimpleVAE, BaseSupervisedVAE):
    """ Supervised VAE where the predictor, encoder and decoder can be built from a list of layers """

    def __init__(self, input_shape, latent_dim, pred_dim, pred_var=0.1, learn_uncertainty=False):
        super(SimpleSupervisedVAE, self).__init__(input_shape=input_shape,
                                                  latent_dim=latent_dim,
                                                  pred_var=pred_var,
                                                  pred_dim=pred_dim,
                                                  learn_uncertainty=learn_uncertainty)
        self.predictorLayers_ = []

    def add_predictor_layer(self, layer):
        """ Append a keras Layer to self.predictorLayers_ """
        self.predictorLayers_.append(layer)

    def build_predictor(self, predict_activation=None):
        """ Construct the predictor network from the list of layers

        After the last layer in self.predictorLayers_, a final Dense layer is added
        that with self.predDim_ units (i.e. outputs the prediction)

        Args:
            predict_activation: activation function for the final dense layer

        """

        if len(self.predictorLayers_) == 0:
            raise ValueError("Must add at least one predictor hidden layer")

        pred_in = self._build_decoder_inputs()
        h = self._edit_decoder_inputs(pred_in)
        for hid in self.predictorLayers_:
            h = hid(h)

        y_pred = Dense(units=self.predDim_,
                       activation=predict_activation)(h)
        log_var_y = Dense(self.predDim_, name='log_var_y')(h)

        if not self.learnUncertainty_:
            log_var_y = Lambda(lambda lv: 0 * lv + K.ones_like(lv) * K.log(K.variable(self.predVar_)))(log_var_y)

        self.predictor_ = Model(inputs=pred_in, outputs=[y_pred, log_var_y], name='predictor')

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        BaseSupervisedVAE.build_vae(self, epsilon_std=epsilon_std, kl_scale=kl_scale)

    def save_all_weights(self, prefix):
        BaseSupervisedVAE.save_all_weights(self, prefix)

    def load_all_weughts(self, prefix):
        BaseSupervisedVAE.load_all_weights(self, prefix)


class SimpleSemiSupervisedVAE(SimpleSupervisedVAE, BaseSemiSupervisedVAE):
    """ Supervised VAE where the predictor, encoder and decoder can be built from a list of layers """

    def __init__(self, input_shape, latent_dim, pred_dim=1., pred_var=0.1, learn_uncertainty=False):
        super(SimpleSupervisedVAE, self).__init__(input_shape=input_shape,
                                                  latent_dim=latent_dim,
                                                  pred_var=pred_var,
                                                  pred_dim=pred_dim,
                                                  learn_uncertainty=learn_uncertainty)

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        BaseSemiSupervisedVAE.build_vae(self, epsilon_std=epsilon_std, kl_scale=kl_scale)

    def save_all_weights(self, prefix):
        BaseSupervisedVAE.save_all_weights(self, prefix)

    def load_all_weughts(self, prefix):
        BaseSupervisedVAE.load_all_weights(self, prefix)


class SimpleConditionalVAE(SimpleVAE, BaseConditionalVAE):

    def __init__(self, input_shape, latent_dim, cond_shape=None):
        super(SimpleConditionalVAE, self).__init__(input_shape=input_shape,
                                                   latent_dim=latent_dim,
                                                   cond_shape=cond_shape)

    def _get_decoder_input(self, z, enc_in):
        return BaseConditionalVAE._get_decoder_input(z, enc_in)

    def _build_encoder_inputs(self):
        x = Input(shape=self.inputShape_)
        c = Input(shape=self.condShape_)
        return [x, c]

    def _build_decoder_inputs(self):
        z = Input(shape=(self.latentDim_,))
        c = Input(shape=self.condShape_)
        return [z, c]

    def _edit_encoder_inputs(self, enc_in):
        x = Flatten()(enc_in[0])
        c = Flatten()(enc_in[1])
        h = Concatenate()[x, c]
        return h

    def _edit_decoder_inputs(self, dec_in):
        z = dec_in[0]
        c = Flatten()(dec_in[1])
        h = Concatenate()[z, c]
        return h


class SimpleSupervisedConditionalVAE(SimpleSupervisedVAE, SimpleConditionalVAE):

    def __init__(self, input_shape, latent_dim, pred_dim=1., pred_var=0.1, cond_shape=None):
        super(SimpleConditionalVAE, self).__init__(input_shape=input_shape,
                                                   latent_dim=latent_dim,
                                                   pred_var=pred_var,
                                                   pred_dim=pred_dim,
                                                   cond_shape=cond_shape)

    def _get_decoder_input(self, z, enc_in):
        return SimpleConditionalVAE._get_decoder_input(z, enc_in)

    def _build_encoder_inputs(self):
        """ BUILD (not get) the encoder inputs """
        return SimpleConditionalVAE._build_encoder_inputs()

    def _build_decoder_inputs(self):
        return SimpleConditionalVAE._build_decoder_inputs()

    def _edit_encoder_inputs(self, enc_in):
        return SimpleConditionalVAE._edit_encoder_inputs(enc_in)

    def _edit_decoder_inputs(self, dec_in):
        return SimpleConditionalVAE._edit_decoder_inputs(dec_in)


class SimpleSemiSupervisedConditionalVAE(SimpleSemiSupervisedVAE, SimpleConditionalVAE):

    def __init__(self, input_shape, latent_dim, pred_dim=1., pred_var=0.1, cond_shape=None):
        super(SimpleConditionalVAE, self).__init__(input_shape=input_shape,
                                                   latent_dim=latent_dim,
                                                   pred_var=pred_var,
                                                   pred_dim=pred_dim,
                                                   cond_shape=cond_shape)

    def _get_decoder_input(self, z, enc_in):
        return SimpleConditionalVAE._get_decoder_input(z, enc_in)

    def _build_encoder_inputs(self):
        """ BUILD (not get) the encoder inputs """
        return SimpleConditionalVAE._build_encoder_inputs()

    def _build_decoder_inputs(self):
        return SimpleConditionalVAE._build_decoder_inputs()

    def _edit_encoder_inputs(self, enc_in):
        return SimpleConditionalVAE._edit_encoder_inputs(enc_in)

    def _edit_decoder_inputs(self, dec_in):
        return SimpleConditionalVAE._edit_decoder_inputs(dec_in)

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        return SimpleSemiSupervisedVAE.build_vae(epsilon_std=epsilon_std, kl_scale=kl_scale)
