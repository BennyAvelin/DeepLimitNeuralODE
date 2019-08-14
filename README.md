# DeepLimitNeuralODE
The code for the paper Neural ODEs as the Deep Limit of ResNets with constant weights, with K. Nyström, 2019, [arXiv](https://arxiv.org/abs/1906.12183)

This repo contains the code to run the finite layer neural ODEs from the paper

* `Annulus_crossval.py` Runs the Annulus experiment, the way to run it is as follows
```./Annulus_crossval.py depth epochs gpus k-fold augmented_dimensions```
* `Cifar10_crossval.py` Runs the Cifar10 experiment, use the call pattern
```./Cifar10_crossval.py depth epochs gpus k-fold```