import autoencodeSVJ.models as models
import autoencodeSVJ.trainer as trainer
import autoencodeSVJ.utils as utils
import argparse
import sys

# searches to perform
N = 2

# learning rates to choose from
learning_rates = [
    0.1,
    0.05,
    0.01,
    0.005,
    0.001,
    0.0005,
    0.0001
]

loss_functions = [
    "mse",
    "mae"
]

metrics = [
    "mse",
    "mae"
]

