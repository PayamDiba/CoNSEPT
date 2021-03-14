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

def _get_TF_revComp(TF):
    ret = np.flip(TF, axis = 0)
    ret = np.flip(ret, axis = 1)

    return ret


def _build_TF_dict(PWM, pseudo_count):
    ret = {}
    df = pd.read_csv(PWM, header=None, index_col=None, sep='\t', names = ['0','1','2','3'])
    df = df.values
    sharedLen = None

    for l in df:
        if l[0][0] == '>':
            tfName = str(l[0][1:])
            lenTF = int(l[1])
            if not sharedLen:
                sharedLen = lenTF
            elif sharedLen != lenTF:
                raise ValueError("TFs must have identical lengths")
            ret[tfName,'+'] = np.zeros((lenTF,4,1,1))
            ret[tfName,'-'] = np.zeros((lenTF,4,1,1))
            baseInd = 0
        elif l[0][0] != '<':
            currBase = [float(b)+float(pseudo_count) for b in l]
            currBase = np.true_divide(currBase, np.sum(currBase))
            ret[tfName,'+'][baseInd,:,0,0] = currBase
            baseInd += 1
        elif l[0][0] == '<':
            ret[tfName,'-'] = _get_TF_revComp(ret[tfName,'+'])

    return ret

def get_motifs(PWM, pseudo_count, tfExp_file):
    tfNames = pd.read_csv(tfExp_file, header=0, index_col=0, sep='\t')
    tfNames = tfNames.index.astype('str')
    tfDict =  _build_TF_dict(PWM, pseudo_count)
    ret = []
    for strand in ['+','-']:
        for currTF in tfNames:
            ret.append(tfDict[currTF,strand])

    ret = np.concatenate(ret, axis = -1)
    return tf.convert_to_tensor(ret, dtype=tf.float32)
