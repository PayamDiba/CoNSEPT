# CoNSEPT (Convolutional Neural Network-based Sequence-to-Expression Prediction Tool)
Saurabh Sinha’s Lab, University of Illinois at Urbana-Champaign [Sinha Lab](https://www.sinhalab.net/sinha-s-home)

## Description
CoNSEPT is a tool to predict gene expression in various cis and trans contexts. Inputs to CoNSEPT are enhancer sequence, transcription factor levels in one or many trans conditions, TF motifs (PWMs), and any prior knowledge of TF-TF interactions.

## Getting Started
CoNSEPT can be installed from PyPI:
```pip install CoNSEPT```

## Preparing Input files
To train CoNSEPT, five input files are required (See ```Example``` directory for examples of input files):

1. Enhancer Sequence (FASTA format):

```
>ENHANCER_1
AACCCA...TTACAAT
>ENHANCER_2
CGGACT...TAACATG
```

2. Gene Expression (tab delimited values):

Rows  TRANS_1 TRANS_2 ... TRANS_N
ENHANCER_1  EXPR  EXPR  EXPR
ENHANCER_2 EXPR EXPR  EXPR

3. Transcription Factor Levels (tab delimited values):

```
Rows  TRANS_1 TRANS_2 ... TRANS_N
FACTOR_1  LEVEL LEVEL ... LEVEL
FACTOR_2  LEVEL LEVEL ... LEVEL
FACTOR_3  LEVEL LEVEL ... LEVEL
```

4. TF-TF Interactions (tab delimited values):

This file specifies any prior knowledge of TF-TF interactions forcing the model to capture them:

```
FACTOR_1  FACTOR_1
FACTOR_1  FACTOR_2
FACTOR_2  FACTOR_3
```

5. Position Count/Weight Matrices (tab delimited values):

Count matrices are recommended. All motifs are required to have the same length. Shorter motifs should be padded with equal count/weight over four nucleotides.

```
>FACTOR_1 LENGTH
A_COUNT C_COUNT G_COUNT T_COUNT
A_COUNT C_COUNT G_COUNT T_COUNT
...
A_COUNT C_COUNT G_COUNT T_COUNT
<
>FACTOR_2 LENGTH
A_COUNT C_COUNT G_COUNT T_COUNT
A_COUNT C_COUNT G_COUNT T_COUNT
...
A_COUNT C_COUNT G_COUNT T_COUNT
<
>FACTOR_3 LENGTH
A_COUNT C_COUNT G_COUNT T_COUNT
A_COUNT C_COUNT G_COUNT T_COUNT
...
A_COUNT C_COUNT G_COUNT T_COUNT
<
```

#### NOTE: Enhancer sequence and gene expression files should have the same ordering of enhancers.
