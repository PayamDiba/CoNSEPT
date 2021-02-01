"""
@author: Payam Dibaeinia
"""


"""
This object scans the sequence with provided TFs and calculate LLR on each position

build the object and just run "scan" method.

TODO: make sure rev complement is done correctly here
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf

class LLR_scanner (object):
    def __init__(self, pathSeqFile, tfDict):
        self.tfNames = list(tfDict.keys())
        self.pathSeqFile_ = pathSeqFile
        self.tfDict_ = tfDict
        self.tfMaxLogScore_ = self.get_TF_maxLogScore(tfDict)

    def get_TF_maxLogScore (self, tfDict):
        scores = []
        for t in self.tfNames:
            currTF = np.log(tfDict[t] + 1e-6)
            scores.append(np.sum(np.max(currTF, axis = 1)))
        return scores

    def read_seq_file (self, seq_file):
        """
        This method reads all the sequences from the sequence file
        """
        df = pd.read_csv(seq_file, header=None, index_col=None, sep='\n')
        df = df.values
        allSeq = []
        seqLabel = []
        self.maxLength = 0

        for l in df:
            currLine = l[0]

            if currLine[0] != '>':
                allSeq.append(currLine)
                self.maxLength = max(self.maxLength, len(currLine))
            else:
                seqLabel.append(currLine[1:])

        self.seq_label_order_ = seqLabel

        return seqLabel, allSeq

    def equalize_length_and_augment(self, seq_list, seq_names, nTrain, training = True):
        """
        The first nEnhancer are agumented (if needed) by padding at the end. It makes sure that the order of
        sequences is the same as their order in the expression file.
        The next enhancers (if any) are short enhancers among the first nTrain that were additionally augmented with
        random paddings for training.
        """
        ret = []
        names = []


        for s,n in zip(seq_list, seq_names):
            currL = len(s)
            if currL < self.maxLength:
                d = self.maxLength - currL
                start = 'N' * 0
                end = 'N' * (d - 0)
                s = start + s + end
                ret.append(s)
                names.append(n)
            else:
                ret.append(s)
                names.append(n)

        if training:
            # Further augment training data by random padding of short sequences
            for s,n in zip(seq_list[:nTrain], seq_names[:nTrain]):
                currL = len(s)
                if currL < self.maxLength:
                    d = self.maxLength - currL
                    nAug = int(min(d+1, 10)) # maximum of 10 random augmentation for a short sequence
                    p = np.random.choice(range(1, d+1), nAug - 1, replace = False).tolist()

                    for ns in p:
                        start = 'N' * ns
                        end = 'N' * (d - ns)
                        sAug = start + s + end
                        ret.append(sAug)
                        names.append(n)


        return ret, np.array(names)

    def encode_seq(self, seq_list):
        """
        seq_list: list of DNA sequence as string of length L
        returns: tensor(N,1,L,4), of one-hot encoded sequences

        4 columns: A, C, G, T
        """
        N = len(seq_list)
        L = len(seq_list[0])
        ret = np.zeros((N,1,L,4))


        for si, s in enumerate(seq_list):
            for bi, b in enumerate(s):
                if b.capitalize() == 'A':
                    ret[si,0,bi,0] = 1
                elif b.capitalize() == 'C':
                    ret[si,0,bi,1] = 1
                elif b.capitalize() == 'G':
                    ret[si,0,bi,2] = 1
                elif b.capitalize() == 'T':
                    ret[si,0,bi,3] = 1

                # to makes sure dummy bases result in zero scores
                # note that we work in LLR format and then convert it to probability
                # if need (by taking exp())
                elif b.capitalize() == 'N':
                    ret[si,0,bi,:] = 1e6


        return tf.convert_to_tensor(ret)

    def segmentize(self, encoded_seq, size):
        """
        segmentize sequence suitable for PWM scanning

        encoded_seq: tensor(N,1,L,4) outputed by encode_seq method
        size: segmentation size (equivalent to PWMs' length)

        returns tensor(N,S,size,4) where S is the number of segments extracted
        """
        N = encoded_seq.shape[0]
        L = encoded_seq.shape[2]
        seg_range = L - size + 1

        ret = [tf.slice(encoded_seq, (0,0,i,0),(N,1,size,4)) for i in range(seg_range)]
        return tf.concat(ret, axis = 1)

    def get_TF_tensor(self, tfDict):
        """
        computes the reverse complement of each TF motif

        returns tensor(4l,2t) where l is the length of TFs and t is number of TFs
        """
        retP = []
        retN = []
        for t in self.tfNames:
            TF = tfDict[t]
            revComp_TF = self.get_TF_revComp(TF)

            TF = np.reshape(TF,(-1,))
            revComp_TF = np.reshape(revComp_TF,(-1,))
            retP.append(TF)
            retN.append(revComp_TF)

        retP = np.array(retP).T
        retN = np.array(retN).T

        ret = np.concatenate((retP, retN), axis = 1)
        return tf.convert_to_tensor(ret)


    def get_TF_revComp(self,TF):
        ret = np.flip(TF, axis = 0)
        ret = np.flip(ret, axis = 1)

        return ret

    def scan(self, nTrain, LLR = False, training = True):
        """
        Scan the sequence to compute matching scores with the TF motifs. Note
        that both strands are in their correct order, so no additional flip is required.
        """

        seqNames, seq_list = self.read_seq_file(self.pathSeqFile_)
        seq_list, seqNames = self.equalize_length_and_augment(seq_list, seqNames, nTrain, training)
        seq_tensor = self.encode_seq(seq_list) #tensor(N,1,L,4)
        TFs_tensor = self.get_TF_tensor(self.tfDict_) #tensor(4l,2t)
        TFs_tensor = tf.math.log(TFs_tensor + 1e-6)
        lenTF = TFs_tensor.shape[0]//4
        segment_tensor = self.segmentize(seq_tensor, size = lenTF) #tensor(N,S,l,4)
        N = segment_tensor.shape[0]
        S = segment_tensor.shape[1]
        segment_tensor = tf.reshape(segment_tensor,(N,S,-1)) #tensor(N,S,4l)

        scores = tf.tensordot(segment_tensor, TFs_tensor, axes = 1) #tensor(N,S,2t)
        maxTF = tf.convert_to_tensor(np.array(2 * self.tfMaxLogScore_))

        if not LLR:
            """
            This mode was used for Sayal et al. data
            """
            scores = tf.exp(scores)
            maxTF = tf.exp(maxTF)
            scores = tf.math.divide(scores,maxTF) #tensor(N,S,2t)

        elif LLR:
            """
            This is very similar (with minor difference) to binding score used by GEMSTAT
            """
            scores = tf.math.divide(scores,maxTF)
            scores = tf.exp(scores) #tensor(N,S,2t)


        scores = tf.reshape(scores,(N,S,2,-1))
        return scores, seqNames
