"""
@author: Payam Dibaeinia
"""

from CoNSEPT.Seq_Scan import Seq
from CoNSEPT.S2E import seq2expr
from CoNSEPT.utils import early_stop, save_pickle, save_expr
from CoNSEPT.utils import make_dir
import numpy as np
import tensorflow as tf
import argparse


## TODO: seperate build_record functionality from training
## TODO: write a "complete.txt" file to TF records path to make sure it was successfully built
## TODO: write predit method
## TODO: write properties and size of data sets to prevent running scanner.scan when reading TF records

def main():

    """
    Define flags
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sf', type=str, default=None, help='path to sequence file containing all train, validation, and test sequences', required = True)
    parser.add_argument('--ef', type=str, default=None, help='path to gene expression file', required = True)
    parser.add_argument('--tf', type=str, default=None, help='path to TF expression file', required = True)
    parser.add_argument('--cf', type=str, default=None, help='path to cooperative interactions file', required = True)
    parser.add_argument('--pwm', type=str, default=None, help='path to PWMs file', required = True)
    parser.add_argument('--pcount', type=float, default=0.0, help='pseudo count for modifying PWMs | Default: 0.0', required = False)
    parser.add_argument('--nb', type=int, default=None, help='number of conditions present in expression files', required = True)
    parser.add_argument('--nTrain', type=int, default=None, help='the first nTrain sequences/genes are used for training', required = True)
    parser.add_argument('--nValid', type=int, default=None, help='the next nValid sequences/genes are used for validation', required = True)
    parser.add_argument('--nTest', type=int, default=None, help='the last nTest sequences/genes are used for testing', required = True)
    parser.add_argument('--psb', type=str, default=None, help='pool size for extract binding sites, specify as x,2 (e.g. 5,2 or 10,2)', required = True)
    parser.add_argument('--csc', type=str, default=None, help='size of kernels capturing prior TF interaction, specify as x,2 (e.g. 5,2 or 10,2)', required = True)
    parser.add_argument('--sc', type=str, default=None, help='stride of kernels capturing prior TF interaction, specify as x,2 (e.g. 5,2 or 10,2)', required = True)
    parser.add_argument('--nChan_noPrior', type=int, default=None, help='number of channels in the layer capturing general TF interactions', required = True)
    parser.add_argument('--nChans', type=str, default=None, help='number of channels in additional layers capturing longer-range interactions, specify one number per layer as x,y,z (e.g. 64,64,16,4 corresponds to four layers)', required = True)
    parser.add_argument('--cAct', type=str, default=None, help='network activation function, is used for all layers before the output', required = True)
    parser.add_argument('--oAct', type=str, default=None, help='output layer activation function | Default: No activation', required = False)
    parser.add_argument('--dr', type=float, default=0.0, help='Network dropout rate | Default: 0', required = False)
    parser.add_argument('--cl2', type=float, default=0.01, help='weight of L2 regularization for TF interaction kernels | Default: 0.01', required = False)
    parser.add_argument('--fcl2', type=float, default=0.01, help='weight of L2 regularization for long range kernels | Default: 0.01', required = False)
    parser.add_argument('--bs', type=int, default=32, help='batch size | Default: 32', required = False)
    parser.add_argument('--nEpoch', type=int, default=100, help='number of training epochs | Default: 100', required = False)
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate of ADAM optimizer | Default: 0.00001', required = False)
    parser.add_argument('--step_LR', type=float, default=None, help='a threshold for train and validation error to control their fluctuations by tuning learning rate. A larger threshhold results in more fluctuations | Default: no threshold', required = False)
    parser.add_argument('--save_freq', type=int, default=4, help='saving frequency | Default: 4', required = False)
    parser.add_argument('--o', type=str, default=None, help='path to output directory', required = True)
    parser.add_argument('--ds', type=str, default='inline', help='data source, options: build_records, read_records, or inline', required = False)
    parser.add_argument('--record_path', type=str, default=None, help='path for reading/writing records. Only needed for build_records and read_records data sources', required = False)
    parser.add_argument('--restore', action='store_true', help='if specified, training is resumed from the last saved epoch', required = False)

    parser.add_argument('--predict', action='store_true', help='if specified, the model only makes predictions without training', required = False)
    parser.add_argument('--ckpt', type=int, default=None, help='the saved checkpoint to be used for prediction (if --predict is specified)', required = False)
    parser.add_argument('--pred_dir', type=str, default=None, help='folder name inside the specified output directory to write predictions (if --predict is specified)', required = False)

    parser.add_argument('--verbose', action='store_true', help='if specified, erros during training epochs are printed to the console (useful when running in jupyter notebook)', required = False)


    #flags.DEFINE_boolean('roles', False,'whether estimate and save TF roles during training')
    #flags.DEFINE_boolean('LLR', False,'whether calulate binding scores similar to GEMSTAT (LLR = True) or just calculate matching probabilities (LLR = False)')

    FLAGS = parser.parse_args()

    """
    Make directories
    """
    make_dir(FLAGS.o)
    make_dir(FLAGS.o + '/checkpoints')
    #make_dir(FLAGS.o + '/TFrecords')
    #if FLAGS.roles:
    #    make_dir(FLAGS.o + '/TF_roles')

    if FLAGS.predict:
        make_dir(FLAGS.o + FLAGS.pred_dir)

    """
    Define model
    """
    model = seq2expr(FLAGS)

    """
    If only require predictions
    """
    if FLAGS.predict:
        model.restore(epoch = FLAGS.ckpt)
        data = Seq(seq_file = FLAGS.sf,
                   PWM = FLAGS.pwm,
                   pseudo_count = FLAGS.pcount,
                   expression_file = FLAGS.ef,
                   TF_file = FLAGS.tf,
                   nBins = FLAGS.nb,
                   nTrain = 0,
                   nValid = 0,
                   nTest = FLAGS.nTest,
                   record_path = FLAGS.record_path,
                   source = FLAGS.ds,
                   LLR = False,
                   training = False)

        pred_expr = np.array([])
        tmp = 0
        for batch in data.batch(type = 'test', nBatch = FLAGS.bs, shuffle = False):
            print(tmp)
            tmp+=1
            currSeq, currTF, _ = (batch['score'], batch['tfE'], batch['gE'])
            currPred = model.predict(seq = currSeq, conc = currTF)
            if pred_expr.shape[0] != 0:
                pred_expr = np.concatenate((pred_expr, currPred), axis = 0)
            else:
                pred_expr = currPred

        pred_expr = np.reshape(pred_expr, (-1,FLAGS.nb))
        save_expr(FLAGS.o + FLAGS.pred_dir +'/predictions_epoch_' + str(FLAGS.ckpt) + '.csv', pred_expr)
        return

    """
    Prepare data
    """
    data = Seq(seq_file = FLAGS.sf,
               PWM = FLAGS.pwm,
               pseudo_count = FLAGS.pcount,
               expression_file = FLAGS.ef,
               TF_file = FLAGS.tf,
               nBins = FLAGS.nb,
               nTrain = FLAGS.nTrain,
               nValid = FLAGS.nValid,
               nTest = FLAGS.nTest,
               record_path = FLAGS.record_path,
               source = FLAGS.ds,
               LLR = False,
               training = not FLAGS.predict)

    print('data is ready')


    """
    Restore model if exist
    """
    if FLAGS.restore:
        epoch = model.restore() + 1
    else:
        epoch = 0



    #train_data = data.dataset(type = 'train', nBatch = FLAGS.bs, shuffle = True)
    #TODO for now I disabled reading the whole training data at once due to possible memory errors
    #TODO however I need to enable prediction on all enhancers without additional agumentation --> should do in batch setting anyways


    """
    Training
    """
    terminate = False
    reduced_LR_count = 0
    pbar = range(FLAGS.nEpoch)
    for epoch in pbar:
        #Train step
        for batch in data.batch(type = 'train', nBatch = FLAGS.bs, shuffle = True):
            currSeq, currTF, currExp = (batch['score'], batch['tfE'], batch['gE'])
            model.train_step(seq = currSeq, conc = currTF, gt_expr = currExp)

        model.collect_loss_train()

        #Valid step
        for batch in data.batch(type = 'valid', nBatch = FLAGS.bs, shuffle = False):
            currSeq, currTF, currExp = (batch['score'], batch['tfE'], batch['gE'])
            model.valid_step(seq = currSeq, conc = currTF, gt_expr = currExp)

        model.collect_loss_valid()

        if FLAGS.verbose == 'True':
            print('Epoch: {}\ttraining loss: {:0.3f}\tvalidation loss:{:0.3f}\tLR:{}'.format(epoch+1,model.loss_train, model.loss_valid, model.optimizer.lr.read_value()))

        # Valid step, do it after each epoch cause we might use it for early termination
        #model.valid_step(seq = valid_seq, conc = valid_TF, gt_expr = valid_exp)

        # save intermediate results if needed
        if (epoch + 1) % FLAGS.save_freq == 0:
            #Test step
            for batch in data.batch(type = 'test', nBatch = FLAGS.bs, shuffle = False):
                currSeq, currTF, currExp = (batch['score'], batch['tfE'], batch['gE'])
                model.test_step(seq = currSeq, conc = currTF, gt_expr = currExp)

            model.collect_loss_test()
            model.save(epoch)

            #if FLAGS.roles:
            #    TF_roles_train = model.compute_TF_roles(seq = train_seq, conc = train_TF)
            #    TF_roles_valid = model.compute_TF_roles(seq = valid_seq, conc = valid_TF)
            #    TF_roles_test = model.compute_TF_roles(seq = test_seq, conc = test_TF)

            #    path_write = FLAGS.o + '/TF_roles/'

            #    save_pickle(path_write + 'train_epoch_' + str(epoch) + '.pkl', TF_roles_train)
            #    save_pickle(path_write + 'valid_epoch_' + str(epoch) + '.pkl', TF_roles_valid)
            #    save_pickle(path_write + 'test_epoch_' + str(epoch) + '.pkl', TF_roles_test)


        # Modify learning rate
        if FLAGS.step_LR:
            model.collect_epoch_loss()
            ## check if training loss was not changed in last 3 steps --> increase LR
            if epoch < 50 and model.nd_train == 3:
                model.scale_LR(10)
                model.nd_train = 0
                model.nd_valid = 0

            ## check if validation loss is increasing in last 3 steps --> reduce LR
            elif model.nd_valid == 3:
                model.scale_LR(0.1)
                model.nd_valid = 0
                model.nd_train = 0

        model.ckpt.step.assign_add(1)
        #For now I disbaled early stopping
        #terminate = early_stop(model.loss_train, model.loss_valid, 0.0061)


    # predict expression after training
    #all_seq, all_TF, all_rho = data.get_data(data = 'all')
    #pred_expr = model.predict(seq = all_seq, conc = all_TF)
    #pred_expr = np.reshape(pred_expr, (-1,FLAGS.nb))
    #save_expr(FLAGS.o + '/predicted_exprs.csv', pred_expr)


if __name__ == "__main__":
    main()
