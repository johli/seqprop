# SeqProp
Stochastic Sequence Propagation - A Keras Model for generating functional RNA based on a predictor

A Python API for constructing generative DNA/RNA Sequence PWM models in Keras. Implements a PWM generator model (with support for discrete sampling with ST gradient estimation), a predictor model wrapper and an optimizer/loss model.

#### Features
- Implements a Sequence PWM Generator as a Keras Model, outputting a sequence PWM, its Logits, and (optionally) random discrete samples from the PWM. These representation can be fed into any downstream Keras model in order to optimize the generator PWM model.
- Implements a Predictor Keras Model wrapper, allowing easy loading of pre-trained sequence models and connecting them to the upstream PWM generator. The Predictor can either be hooked to the continuous PWM output, or the discerete sample One-hot sequences.
- Implements a Loss model with various useful loss functions and objectives, including common sequence Predictor losses and regularizing PWM losses (e.g., soft sequence constraints, PWM entropy costs, etc.)
- Includes visualization code for plotting Sequence PWMs and various cost functions during optimization (as Keras Callbacks).




