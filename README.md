# SeqProp
Stochastic Sequence Propagation - A Keras Model for generating functional RNA based on a predictor

A Python API for constructing generative DNA/RNA Sequence PWM models in Keras. Implements a PWM generator (with support for discrete sampling with ST gradient estimation), a predictor model wrapper and a loss model.

#### Features
- Implements a Sequence PWM Generator as a Keras Model, outputting PWMs, Logits, or random discrete samples from the PWM. These representations can be fed into any downstream Keras model for reinforcement learning.
- Implements a Predictor Keras Model wrapper, allowing easy loading of pre-trained sequence models and connecting them to the upstream PWM generator.
- Implements a Loss model with various useful cost and objectives, including regularizing PWM losses (e.g., soft sequence constraints, PWM entropy costs, etc.)
- Includes visualization code for plotting PWMs and cost functions during optimization (as Keras Callbacks).

### Installation
SeqProp can be installed by cloning or forking the [github repository](https://github.com/johli/seqprop.git):
```sh
git clone https://github.com/johli/seqprop.git
cd seqprop
python setup.py install
```

#### SeqProp requires the following packages to be installed
- Tensorflow >= 1.13.1
- Keras >= 2.2.4
- Scipy >= 1.2.1
- Numpy >= 1.16.2

### Usage
Isolearn is centered around data generators, where the generator's task is to transform your sequence data (stored in a Pandas dataframe) and corresponding measurements (e.g. column in the Pandas dataframe, or RNA-Seq count matrix) into numerical input features and output targets.

A simple Keras Data Generator can built using the isolearn.keras package:
```python
import isolearn.keras as iso
import pandas as pd
import numpy as np

```

### Example Notebooks (Alternative Polyadenylation)
These examples show how to set up the PWM sequence generator model, hooking it up to a predictor, and defining various loss models. The examples all build on the Alternative Polyadenylation sequence predictor APARENT.

[Notebook 1a: Generate Target Isoforms (Predict on PWM)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization.ipynb)<br/>
[Notebook 1b: Generate Target Isoforms (Predict on Sampled One-hots)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization_sample.ipynb)<br/>
[Notebook 2: Generate Target 3' Cleavage (Predict on Sampled One-hots)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_cleavage_optimization.ipynb)<br/>
