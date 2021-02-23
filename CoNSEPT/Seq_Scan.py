"""
@author: Payam Dibaeinia
"""

import numpy as np
import pandas as pd
from CoNSEPT.Scanners import LLR_scanner
import tensorflow as tf
from CoNSEPT.DataBuilder import TFRecordData, InlineData

class Seq(object):
    def __init__(self,
                 seq_file,
                 PWM,
                 pseudo_count,
                 expression_file,
                 TF_file,
                 nBins,
                 nTrain,
                 nValid,
                 nTest,
                 record_path,
                 source,
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

        self.rPath_ = record_path
        self.source_ = source
        tf_motif_dict = self.build_TF_dict(PWM, pseudo_count)
        scanner = LLR_scanner(seq_file, tf_motif_dict)
        tfExpr = self.read_TF_file(TF_file, tf_order = scanner.tfNames)
        seqExpr = self.read_expr_file(expression_file)
        scores, scoreLabels = scanner.scan(nTrain, LLR, training)

        if scanner.seq_label_order_ != list(seqExpr.index):
            raise ValueError('Input files are inconsistent, use the same order for sequences in the input sequence and expression files')


        if tfExpr.shape[1] != nBins or seqExpr.shape[1] != nBins:
            raise ValueError('Input files are inconsistent, tf or gene expression files have different number of conditions than nBins')



        if source == "build_records":
            self.data_ = TFRecordData()
            self._build_record(scores, scoreLabels, seqExpr, tfExpr, nBins, nTrain, nValid, nTest, LLR, training, record_path)
        elif source == "read_records":
            self.data_ = TFRecordData()
            trainLabels, scores_train, scores_train_noAug, scores_valid, scores_test = self._split_data(scores, nTrain, nValid, nTest, list(seqExpr.index)[:nTrain], scoreLabels)
            scores = None
            self.score_shape = scores_train.shape[1:]
            self.tfE_shape = tfExpr.shape[0]
            self.trainSize = scores_train.shape[0] * nBins
            self.validSize = scores_valid.shape[0] * nBins
            self.testSize = scores_test.shape[0] * nBins
        elif source == "inline":
            trainLabels, scores_train, scores_train_noAug, scores_valid, scores_test = self._split_data(scores, nTrain, nValid, nTest, list(seqExpr.index)[:nTrain], scoreLabels)
            scores = None
            self.data_ = InlineData(trainLabels, scores_train, scores_valid, scores_test, seqExpr, tfExpr, nTrain, nValid, nTest, nBins)
            self.score_shape = scores_train.shape[1:]
            self.tfE_shape = tfExpr.shape[0]
            self.trainSize = scores_train.shape[0] * nBins
            self.validSize = scores_valid.shape[0] * nBins
            self.testSize = scores_test.shape[0] * nBins

    def _build_record(self, scores, scoreLabels, seqExpr, tfExpr, nBins, nTrain, nValid, nTest, LLR, training, record_path):
        trainLabels, scores_train, _, scores_valid, scores_test = self._split_data(scores, nTrain, nValid, nTest, list(seqExpr.index)[:nTrain], scoreLabels)
        scores = None

        print("start building records")
        self.data_.build_record(record_path + '/train.tfrecord', scores_train.numpy(), seqExpr.iloc[:nTrain], tfExpr, score_labels = trainLabels)
        print("done building train records")
        self.data_.build_record(record_path + '/valid.tfrecord', scores_valid.numpy(), seqExpr.iloc[nTrain:nTrain+nValid], tfExpr)
        print("done building valid records")
        self.data_.build_record(record_path + '/test.tfrecord', scores_test.numpy(), seqExpr.iloc[nTrain+nValid:], tfExpr)
        print("done building test records")
        self.score_shape = scores_train.shape[1:]
        self.tfE_shape = tfExpr.shape[0]
        self.trainSize = scores_train.shape[0] * nBins
        self.validSize = scores_valid.shape[0] * nBins
        self.testSize = scores_test.shape[0] * nBins

    def _split_data(self, scores, nTrain, nValid, nTest, trainLabels, scoreLabels):
        indTrain = np.where(np.in1d(scoreLabels, trainLabels))[0]
        labels = scoreLabels[indTrain]

        scores_train = tf.gather(scores, indTrain, axis = 0)
        scores_valid = scores[nTrain:nTrain + nValid]
        scores_test = scores[nTrain+nValid: nTrain+nValid+nTest]
        scores_train_noAug = scores[:nTrain]

        return labels, scores_train, scores_train_noAug, scores_valid, scores_test


    def build_TF_dict(self, PWM, pseudo_count):
        ret = {}
        df = pd.read_csv(PWM, header=None, index_col=None, sep='\t', names = ['0','1','2','3'])
        df = df.values
        sharedLen = None

        for l in df:
            if l[0][0] == '>':
                tfName = l[0][1:]
                lenTF = int(l[1])
                if not sharedLen:
                    sharedLen = lenTF
                elif sharedLen != lenTF:
                    raise ValueError("TFs must have identical lengths")
                ret[tfName] = np.zeros((lenTF,4))
                baseInd = 0
            elif l[0][0] != '<':
                currBase = [float(b)+float(pseudo_count) for b in l]
                currBase = np.true_divide(currBase, np.sum(currBase))
                ret[tfName][baseInd,:] = currBase
                baseInd += 1

        return ret


    def read_expr_file(self, expr_file):
        df = pd.read_csv(expr_file, header=0, index_col=0, sep='\t')
        return df

    def read_TF_file(self, TF_file, tf_order):
        df = pd.read_csv(TF_file, header=0, index_col=0, sep='\t')
        currInds = list(df.index)
        inds = [currInds.index(i) for i in tf_order]
        df = df.iloc[inds,:]

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
                nBatch = self.trainSize

            if self.source_ == 'inline':
                return self.data_.get_dataset(type, shuffle, 10*self.trainSize, nBatch)
            else:
                return self.data_.get_dataset(self.rPath_ + '/train.tfrecord', self.score_shape, self.tfE_shape, nBatch, shuffle)

        elif type == 'valid':
            if nBatch == -1:
                nBatch = self.validSize

            if self.source_ == 'inline':
                return self.data_.get_dataset(type, False, 1000, nBatch)
            else:
                return self.data_.get_dataset(self.rPath_ + '/valid.tfrecord', self.score_shape, self.tfE_shape, nBatch, False)

        elif type == 'test':
            if nBatch == -1:
                nBatch = self.testSize

            if self.source_ == 'inline':
                return self.data_.get_dataset(type, False, 1000, nBatch)
            else:
                return self.data_.get_dataset(self.rPath_ + '/test.tfrecord', self.score_shape, self.tfE_shape, nBatch, False)

    # def get_batches(self):
    #     seqScores = []
    #     gExpr = []
    #     tfExpr = []
    #
    #     for bInd in list(self.trainID.as_numpy_iterator()):
    #         scoreIDs = bInd//self.nBins_
    #         binIDs = bInd%self.nBins_
    #         exprIDs = self.map_[scoreIDs]
    #         currGE = []
    #         currTFE = []
    #
    #         for bi, ei in zip(binIDs, exprIDs):
    #             currGE.append(self.seqExpr.values[ei,bi])
    #             currTFE.append(self.tfExpr.values[:,bi].tolist())
    #
    #         gExpr.append(np.array(currGE).reshape(-1,1))
    #         tfExpr.append(np.array(currTFE))
    #         seqScores.append(tf.gather(self.scores_train, scoreIDs, axis = 0))
    #
    #     return seqScores, tfExpr, gExpr
    #
    #
    # def get_data(self, data = 'all'):
    #
    #     """
    #     data: 'train', 'test', 'valid', or 'all'
    #     """
    #     if data == 'train':
    #         N = self.scores_train_noAug.shape[0] #number of sequences
    #         nTF = self.tfExpr.values.shape[0]
    #         retScores = tf.tile(self.scores_train_noAug, [self.nBins_,1,1,1])
    #
    #         retTFE = np.tile(self.tfExpr.values, [N,1]).reshape((-1,1), order='F').flatten()
    #         retTFE = np.array(np.array_split(retTFE, len(retTFE)//nTF))
    #
    #         retGE = self.seqExpr.values[:self.nTrain,:].reshape((-1,1), order='F')
    #
    #         return retScores, retTFE, retGE
    #
    #     elif data == 'valid':
    #         N = self.scores_valid.shape[0] #number of sequences
    #         nTF = self.tfExpr.values.shape[0]
    #         retScores = tf.tile(self.scores_valid, [self.nBins_,1,1,1])
    #
    #         retTFE = np.tile(self.tfExpr.values, [N,1]).reshape((-1,1), order='F').flatten()
    #         retTFE = np.array(np.array_split(retTFE, len(retTFE)//nTF))
    #
    #         retGE = self.seqExpr.values[self.nTrain:self.nTrain+self.nValid,:].reshape((-1,1), order='F')
    #
    #         return retScores, retTFE, retGE
    #
    #     elif data == 'test':
    #         N = self.scores_test.shape[0] #number of sequences
    #         nTF = self.tfExpr.values.shape[0]
    #         retScores = tf.tile(self.scores_test, [self.nBins_,1,1,1])
    #
    #         retTFE = np.tile(self.tfExpr.values, [N,1]).reshape((-1,1), order='F').flatten()
    #         retTFE = np.array(np.array_split(retTFE, len(retTFE)//nTF))
    #
    #         retGE = self.seqExpr.values[self.nTrain+self.nValid:,:].reshape((-1,1), order='F')
    #
    #         return retScores, retTFE, retGE
    #
    #     elif data == 'all':
    #         trainScore, trainTFE, trainGE = self.get_data('train')
    #         validScore, validTFE, validGE = self.get_data('valid')
    #         testScore, testTFE, testGE = self.get_data('test')
    #
    #         retScores = tf.concat((trainScore, validScore, testScore), axis = 0)
    #         retTFE = np.concatenate((trainTFE, validTFE, testTFE), axis = 0)
    #         retGE = np.concatenate((trainGE, validGE, testGE), axis = 0)
    #         return retScores, retTFE, retGE
