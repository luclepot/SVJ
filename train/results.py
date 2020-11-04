import time
import datetime
import autoencodeSVJ.utils as utils
import autoencodeSVJ.evaluate as ev
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import OrderedDict as odict
import pandas as pd
import glob
import os
import tensorflow as tf

TRAIN = True

lr = .00051
lr_factor = 0.5
es_patience = 12
target_dim = 8
batch_size = 32
norm_percentile = 25
epochs = 100
n_models = 150 - 114              # number of models to train
model_acceptance_fraction = 10 # take top N best performing models

start_stamp = time.time()
res = None
if TRAIN:
    for i in range(n_models):
        mse = ev.ae_train(
            signal_path='data/all_signals/2000GeV_0.15/base_3/*.h5',
            qcd_path='data/background/base_3/*.h5',
            target_dim=target_dim,
            verbose=False,
            batch_size=batch_size,
            learning_rate=lr,
            norm_percentile=norm_percentile,
            lr_factor=lr_factor,
            es_patience=es_patience,
            epochs=epochs
        )
        print('model {} finished (mse = {:.4f})'.format(i, mse))
        print
    
    res = utils.summary()
    res = res[pd.DatetimeIndex(res.start_time) > datetime.datetime.fromtimestamp(start_stamp)]

else:
    res = utils.summary()
    res = res.sort_values('start_time').tail(n_models)

# take lowest 10% losses of all trainings
n_best = int(0.01*model_acceptance_fraction*n_models)
best_ = res.sort_values('total_loss').head(n_best)
best_name = str(best_.filename.values[0])
