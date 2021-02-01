"""
@author: Payam Dibaeinia
"""

import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf

def early_stop (train_loss, valid_loss, threshold):
    """
    stop training if train_loss < threshold and the difference between train and
    valid loss is small enough
    """

    if train_loss < threshold and np.abs(train_loss - valid_loss) < 0.03 * train_loss:
        return True

    return False

def save_pickle (path_write, array):
    with open(path_write,'wb') as f:
        pickle.dump(array, f)

def save_expr (path_write, array):
    df = pd.DataFrame(array)
    df.to_csv(path_write, index = None, header = None)

def make_dir(path):
    """
    Note that it takes path to a directory not to a writable file
    """

    if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(path)
