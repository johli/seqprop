# SeqProp
Stochastic Sequence Propagation - A Keras Model for generating functional RNA based on a predictor

A Python API for constructing generative DNA/RNA Sequence PWM models in Keras. Implements a PWM generator (with support for discrete sampling with ST gradient estimation), a predictor model wrapper and a loss model.

#### Features
- Implements a Sequence PWM Generator as a Keras Model, outputting PWMs, Logits, or random discrete samples from the PWM. These representations can be fed into any downstream Keras model for reinforcement learning.
- Implements a Predictor Keras Model wrapper, allowing easy loading of pre-trained sequence models and connecting them to the upstream PWM generator.
- Implements a Loss model with various useful cost and objectives, including regularizing PWM losses (e.g., soft sequence constraints, PWM entropy costs, etc.)
- Includes visualization code for plotting PWMs and cost functions during optimization (as Keras Callbacks).




