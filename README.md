![SeqProp Logo](https://github.com/johli/seqprop/blob/master/SeqProp_Logo.jpg?raw=true)

# SeqProp
Stochastic Sequence Propagation - A Keras Model for optimizing DNA, RNA and protein sequences based on a predictor.

A Python API for constructing generative DNA/RNA/protein Sequence PWM models in Keras. Implements a PWM generator (with support for discrete sampling and ST gradient estimation), a predictor model wrapper and a loss model.

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
- Isolearn >= 0.2.0 ([github](https://github.com/johli/isolearn.git))

### Usage
SeqProp provides API calls for building PWM generators and downstream sequence predictors as Keras Models.

A simple generator pipeline for some (imaginary) predictor can be built as follows:
```python
import keras
from keras.models import Sequential, Model, load_model
import isolearn.keras as iso
import numpy as np

from seqprop.visualization import *
from seqprop.generator import *
from seqprop.predictor import *
from seqprop.optimizer import *

from my.project import load_my_predictor #Function that loads your predictor

#Define Loss Function (Fit predicted output to some target)
#Also enforce low PWM entropy

target = np.zeros((1, 1))
target[0, 0] = 5.6 (Arbitrary target)

pwm_entropy_mse = get_target_entropy_sme(pwm_start=0, pwm_end=100, target_bits=1.8)

def loss_func(predictor_outputs) :
  pwm_logits, pwm, sampled_pwm, predicted_out = predictor_outputs
  
  #Create target constant
  target_out = K.tile(K.constant(target), (K.shape(sampled_pwm)[0], 1))
  
  target_cost = (target_out - predicted_out)**2
  pwm_cost = pwm_entropy_mse(pwm)
  
  return K.mean(target_cost + pwm_cost, axis=-1)

#Build Generator Network
_, seqprop_generator = build_generator(seq_length=100, n_sequences=1, batch_normalize_pwm=True)

#Build Predictor Network and hook it on the generator PWM output tensor
_, seqprop_predictor = build_predictor(seqprop_generator, load_my_predictor(), n_sequences=1, eval_mode='pwm')

#Build Loss Model (In: Generator seed, Out: Loss function)
_, loss_model = build_loss_model(seqprop_predictor, loss_func)

#Specify Optimizer to use
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#Compile Loss Model (Minimize self)
loss_model.compile(loss=lambda true, pred: pred, optimizer=opt)

#Fit Loss Model
loss_model.fit([], np.ones((1, 1)), epochs=1, steps_per_epoch=1000)

#Retrieve optimized PWMs and predicted (optimized) target
_, optimized_pwm, _, predicted_out = seqprop_predictor.predict(x=None, steps=1)

```

### Example Notebooks
These examples show how to set up the sequence optimization model, hook it to a predictor, and define various loss models. The examples build on different DNA, RNA and protein design tasks using a wide selection of fitness predictors: APARENT [(Bogard et. al., 2019)](https://doi.org/10.1016/j.cell.2019.04.046), Optimus 5' [(Sample et. al., 2019)](https://doi.org/10.1038/s41587-019-0164-5), DragoNN [(Kundaje Lab)](https://github.com/kundajelab/dragonn), MPRA-DragoNN [(Movva et. al., 2019)](https://doi.org/10.1371/journal.pone.0218073), DeepSEA [(Zhou et. al., 2015)](https://doi.org/10.1038/nmeth.3547) and trRosetta [(Yang et. al., 2020)](https://doi.org/10.1073/pnas.1914677117).

#### Alternative Polyadenylation (APARENT)
[Notebook 1a: Generate Target Isoforms (Predict on PWM)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization.ipynb)<br/>
[Notebook 1b: Generate Target Isoforms (Predict on Sampled One-hots)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization_sample.ipynb)<br/>
[Notebook 2: Generate Target 3' Cleavage (Predict on Sampled One-hots)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_cleavage_optimization.ipynb)<br/>
[Notebook 3a: Evaluate Logit-Normalization](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization_eval_instancenorm_earthmover.ipynb)<br/>
[Notebook 3b: Evaluate Logit-Normalization (Different Gradient Estimators)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization_eval_instancenorm_earthmover_gradient_estimators.ipynb)<br/>
[Notebook 3c: Evaluate Logit-Normalization (Gumbel Sampler)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization_eval_instancenorm_earthmover_vs_evolution_and_gumbel.ipynb)<br/>
[Notebook 3d: Evaluate Logit-Normalization (Explicit Entropy Penalty)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization_eval_instancenorm_earthmover_w_entropy_penalty.ipynb)<br/>
[Notebook 3e: Evaluate Logit-Normalization (Optimizer Settings)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/apa/seqprop_aparent_isoform_optimization_eval_instancenorm_sgd_lr_earthmover.ipynb)<br/>

#### Basic (Pretend-predictor)
[Notebook 1: Apply Sequence Transforms Before Predictor](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/basic/seqprop_basic_sequence_transform.ipynb)<br/>

#### Translational Efficiency (Optimus 5')
[Notebook 1: Evaluate Logit-Normalization](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/optimus5/seqprop_optimus5_optimization_eval_instancenorm_earthmover_non_retrained.ipynb)<br/>

#### CTCF TF Binding (DeepSEA, Dnd41)
[Notebook 1: Evaluate Logit-Normalization](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/deepsea/seqprop_deepsea_optimization_eval_instancenorm_earthmover.ipynb)<br/>

#### Transcriptional Activity (MPRA-DragoNN, SV40, Mean Activity)
[Notebook 1: Evaluate Logit-Normalization](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/mpradragonn/seqprop_mpradragonn_optimization_earthmover_k562_sv40_promoter_deep_factorized_model_eval_instancenorm.ipynb)<br/>

#### SPI1 TF Binding (DragoNN)
[Notebook 1a: Evaluate Logit-Normalization](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/dragonn/seqprop_dragonn_optimization_eval_instancenorm_earthmover.ipynb)<br/>
[Notebook 1b: Evaluate Logit-Normalization (Different Gradient Estimator)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/dragonn/seqprop_dragonn_optimization_eval_instancenorm_earthmover_gradient_estimators.ipynb)<br/>
[Notebook 1c: Evaluate Logit-Normalization (Gumbel Sampler)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/dragonn/https://github.com/johli/seqprop/blob/master/examples/dragonn/seqprop_dragonn_optimization_eval_instancenorm_earthmover_vs_evolution_and_gumbel.ipynb)<br/>
[Notebook 1d: Evaluate Logit-Normalization (Vs. Simulated Annealing)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/dragonn/seqprop_dragonn_optimization_eval_instancenorm_earthmover_vs_evolution_and_basinhopping.ipynb)<br/>

#### Target Protein Structure (trRosetta)
[Notebook 1a: Kinase Protein (No MSA)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/rosetta/seqprop_rosetta_optimization_eval_layernorm_and_basinhopping_T1001_no_msa_1000_updates_multiple_seeds.ipynb)<br/>
[Notebook 1b: Coiled-Coil Hairpin (No MSA)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/rosetta/seqprop_rosetta_optimization_eval_layernorm_and_basinhopping_TR005257_no_msa_1000_updates_multiple_seeds.ipynb)<br/>
[Notebook 2a: Kinase Protein (With MSA)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/rosetta/seqprop_rosetta_optimization_eval_layernorm_and_basinhopping_T1001_with_msa_1000_updates.ipynb)<br/>
[Notebook 2b: Coiled-Coil Hairpin (With MSA)](https://nbviewer.jupyter.org/github/johli/seqprop/blob/master/examples/rosetta/seqprop_rosetta_optimization_eval_layernorm_and_basinhopping_TR005257_with_msa_1000_updates.ipynb)<br/>
