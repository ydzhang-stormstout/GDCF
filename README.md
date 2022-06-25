Geometric Disentangled Collaborative Filtering
================================

The code is built upon https://github.com/oskopek/mvae

## Requirements

- Python 3.7
- Pytorch 1.6.0
- numpy
- scikit-learn
- geoopt

## Usage

### ```main.py```

This script trains models for recommendation. Metrics are printed at the end of training.

```
optional arguments:
  -h, --help            Show this help message and exit
  --dataset 			Name of dataset
  --lr                  Learning rate
  --rg 					L2 regularization
  --dropout 			Dropout probability
  --cuda 	            Which cuda device to use (-1 for cpu training)
  --epoch		        Number of epochs to train 
  --batch				Training batch size
  --dim					Dimension of each embedding
  --manifold			Which manifold to use, an be any on [unite,...]
  --seed                Seed for training
  --beta			    Strength of disentanglement
  --tau   				Temperature of sigmoid/softmax, in (0, 1)
  --std                 Standard deviation of the Gaussian prior
  --nogb                Disable Gumbel-Softmax sampling
  --gamma GAMMA         gamma for lr scheduler
  --component			List of manifold to be used. e: Euclidean, s: hypersphere, h: hyperboloid, e.g.,  'h6'
```