## CoNSEPT (Convolutional Neural Network-based Sequence-to-Expression Prediction Tool)
Saurabh Sinha’s Lab, University of Illinois at Urbana-Champaign [Sinha Lab](https://www.sinhalab.net/sinha-s-home)

## Description
CoNSEPT is a tool to predict gene expression in various cis and trans contexts. Inputs to CoNSEPT are enhancer sequence, transcription factor levels in one or many trans conditions, TF motifs (PWMs), and any prior knowledge of TF-TF interactions.

## Note
Please see [here](https://github.com/PayamDiba/Manuscript_tools/tree/main/CoNSEPT_rho) for the version we used in modeling the regulation of rhomboid gene in Drosophila.

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

```
Rows  TRANS_1 TRANS_2 ... TRANS_N
ENHANCER_1  EXPR  EXPR  EXPR
ENHANCER_2 EXPR EXPR  EXPR
```

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

## Usage
Get a quick help on the required flags for running CoNSEPT:

```consept -h```

Here is the list of arguments for running in training/prediction modes:

* --sf: path to sequence file containing all train, validation, and test sequences  
* --ef: path to gene expression file  
* --tf: path to TFs' level file  
* --cf: path to TF-TF interactions file  
* --pwm: path to PWM/PCM file  
* --pcount: pseudo count for modifying PWMs | Default: 0.0  
* --nb: number of trans conditions present in gene (and TF) expression files  
* --nTrain: the first nTrain sequences (in all trans conditions) are used for training  
* --nValid: the next nValid sequences (in all trans conditions) are used for validation  
* --nTest: the last nTest sequences (in all trans conditions) are used for testing  
* --psb: pool size for extract binding sites, specify as x,2 (e.g. 5,2 or 10,2)  
* --csc: size of kernels capturing prior TF interaction, specify as x,2 (e.g. 5,2 or 10,2)  
* --sc: stride of kernels capturing prior TF interaction, specify as x,2 (e.g. 5,2 or 10,2)  
* --nChan_noPrior: number of channels in the convolutional layer capturing general TF interactions  
* --nChans: number of channels in the additional layers capturing longer-range interactions, specify one number per layer as x,y,z (e.g. 64,64,16,4 corresponds to four additional convolutional layers)  
* --cAct: network activation function, is used for all layers before the output (e.g. relu)  
* --oAct: output layer activation function | Default: No activation  
* --dr: network dropout rate | Default: 0  
* --cl2: weight of L2 regularization for short-range kernels | Default: 0.01  
* --fcl2: weight of L2 regularization for long-range kernels | Default: 0.01  
* --bs: batch size (Note: a training data of n1 enhancers and n2 trans conditions has a total size of n1*n2)| Default: 32  
* --nEpoch: number of training epochs | Default: 100  
* --lr: initial learning rate of ADAM optimizer | Default: 0.00001  
* --step_LR: a threshold for train and validation error to control their fluctuations by tuning learning rate. A larger threshhold results in more fluctuations | Default: no threshold  
* --save_freq: saving frequency (in epochs) | Default: 4  
* --o: path to the output directory  
* --restore: if specified, training is resumed from the last saved epoch  
* --predict: if specified, the model only makes predictions with no training  
* --ckpt: [if --predict is specified] the saved checkpoint to be used for prediction  
* --pred_dir: [if --predict is specified] folder name inside the output directory to write predictions  


## Example
We provided an example dataset in ```example``` directory (data obtained from [here](https://elifesciences.org/articles/08445)). The data consists of expressions driven by 52 enhancers in 17 trans conditions regulated by three TFs. Here is an example command-line for training a CoNSEPT model on this data (38 enhancers are used for training, 3 for validation and the last 11 enhancers for testing):

```consept --sf {PATH-TO}/example/seq.fa --ef {PATH-TO}/example/expr.tab --tf {PATH-TO}/example/factor_expr.tab --cf {PATH-TO}/example/coop.tab --pwm {PATH-TO}/example/PWMs.txt --nb 17 --nTrain 38 --nValid 3 --nTest 11 --psb 4,2 --csc 4,2 --sc 4,2 --nChan_noPrior 0 --nChans 36,6 --cAct relu --oAct sigmoid --dr 0.5 --o {PATH-TO}/outputs```
