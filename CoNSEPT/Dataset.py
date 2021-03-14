"""
@author: Payam Dibaeinia
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from CoNSEPT.DataBuilder import InlineData
import os

class dataset(object):
    def __init__(self,
                 seq_file,
                 expression_file,
                 TF_file,
                 nBins,
                 nTrain,
                 nValid,
                 nTest,
                 out_path,
                 LLR = False,
                 training = True):
        """
        seq_file: path to A FASTA formatted sequence file
        expression_file: path to expression file of each enhancer in one or more conditions/bins
        TF_file: path to expression file of TFs in the same conditions of expression_file
        PWM: path to the PWM file

        nTrain: first nTrain enhancers are used for training
        nValid: next nValid enhancers are used for validation
        nTest: last nTest enhancers are used for testing

        nBins: number of conditions in the expression and TF file
        training: if False, all data is scanned with no offset, otherwise only test data is scanned with no offset. It can
        be considered as an augmentation method.
        """

        self.nTrain_ = nTrain
        self.nTest_ = nTest
        self.nValid_ = nValid
        self.nBins_ = nBins
        self.tfExpr = self.read_TF_file(TF_file)
        seqExpr = self.read_expr_file(expression_file)
        seq_names, max_length, aug = self._read_seq(seq_file)


        if seq_names != list(seqExpr.index):
            raise ValueError('Input files are inconsistent, use the same order for sequences in the input sequence and expression files')


        if self.tfExpr.shape[1] != nBins or seqExpr.shape[1] != nBins:
            raise ValueError('Input files are inconsistent, tf or gene expression files have different number of conditions than nBins')

        if aug: # Note that augmentation updates nTrain and seq_names
            self.nTrain_, seq_file, seq_names = self._augment_data(seq_file, max_length, nTrain, out_path)

        self.data_ = InlineData(seq_file, seq_names, seqExpr, self.tfExpr, self.nTrain_, nValid, nTest, nBins)

    def _read_seq(self, seq_file):
        """ Reads sequences, extracts sequences names and the max length. Also determines wether augmentation/padding is needed

        seq_file: input sequence file
        return: seq_names, max_length, augmentation
        """

        seq_names = []
        max_length = 0
        aug = False

        with open(seq_file,'r') as f:
            rows = f.readlines()

            for currR in rows:
                r = currR.split('\n')[0]
                if r[0] == ">":
                    seq_names.append(r[1:])
                else:
                    currLen = len(r)
                    if aug == False and max_length != 0 and currLen != max_length:
                        aug = True
                    max_length = max(max_length, currLen)

        return seq_names, max_length, aug

    def _augment_data(self, seq_file, max_length, nTrain, path):
        """
        equalizes all sequnece lenghts and augment training sequences
        """

        seq_names = []
        seqPath = path + '/seq_augmented.fa'

        if os.path.exists(seqPath):
            os.remove(seqPath)

        with open(seq_file,'r') as fr:
            rows = fr.readlines()
            with open(seqPath,'w') as fw:

                # Equalize and augment training sequences when needed
                nAugTrain = 0
                for currR in rows[:2*nTrain]:
                    r = currR.split('\n')[0]
                    if r[0] == ">":
                        name = r
                        continue
                    elif len(r) < max_length:
                        currSeq = self._aug(r, max_length)
                    else:
                        currSeq = [r]

                    for s in currSeq:
                        nAugTrain += 1
                        fw.write(name+'\n')
                        fw.write(s+'\n')
                        seq_names.append(name[1:])

                # Equalize remaining sequences when needed
                for currR in rows[2*nTrain:]:
                    r = currR.split('\n')[0]
                    if r[0] == ">":
                        name = r
                        continue
                    elif len(r) < max_length:
                        currSeq = self._equalize(r, max_length)
                    else:
                        currSeq = r

                    fw.write(name+'\n')
                    fw.write(currSeq+'\n')
                    seq_names.append(name[1:])

        return nAugTrain, seqPath, seq_names

    def _aug(self,seq, max_length, nAug = 10):
        ret = []

        d = max_length - len(seq)
        start = 'N' * 0
        end = 'N' * (d - 0)
        s = start + seq + end
        ret.append(s) # Make sure that one augmentation placing the short sequence at the beginning exists in data

        nAug = int(min(d+1, nAug)) #do additional augmentations
        p = np.random.choice(range(1, d+1), nAug - 1, replace = False).tolist()

        for ns in p:
            start = 'N' * ns
            end = 'N' * (d - ns)
            sAug = start + seq + end
            ret.append(sAug)

        return ret

    def _equalize(self, seq, max_length):
        d = max_length - len(seq)
        start = 'N' * 0
        end = 'N' * (d - 0)
        s = start + seq + end
        return s


    def read_expr_file(self, expr_file):
        df = pd.read_csv(expr_file, header=0, index_col=0, sep='\t')
        df.index = df.index.astype('str')
        return df

    def read_TF_file(self, TF_file):
        df = pd.read_csv(TF_file, header=0, index_col=0, sep='\t')
        df.index = df.index.astype(str)
        return df


    def batch(self, type, nBatch = -1, shuffle = True):
        """
        returns tensorflow datasets

        type: 'train', 'test', 'valid'
        nBatch: batch size, if -1 all data is returned
        shuffle: wether to shuffle data after each epoch
        """

        if type == 'train':
            if nBatch == -1:
                nBatch = self.nTrain_ * self.nBins_

            return self.data_.get_dataset(type, shuffle, nBatch)

        elif type == 'valid':
            if nBatch == -1:
                nBatch = self.nValid_ * self.nBins_

            return self.data_.get_dataset(type, False, nBatch)

        elif type == 'test':
            if nBatch == -1:
                nBatch = self.nTest_ * self.nBins_

            return self.data_.get_dataset(type, False, nBatch)
