#! /usr/bin/env python

import argparse
import os
import logging
import datetime
import glob
import torch
import numpy as np
import pandas as pd
import torchtext.data as data
import models
from models import model_config
import train
from data_processing import DATA_PATH, CLASSES
from data_processing import tokenizer, create_submission
from data_processing import make_iterators, ToxicComments

parser = argparse.ArgumentParser(description='Toxic Comments Classification')

# processing params
parser.add_argument('-token-threshold', type=int, default=1,
                    help='threshold for a token frequency to be considered in vocabulary'
                         'tokens below threshold will be <unk>')
parser.add_argument('-comment-size', type=int, default=200,
                    help='fixed size of comment in terms of number of tokens')
# training params
parser.add_argument('-folds', type=int, default=10, help='number of folds for cross-validation')
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=15, help='number of epochs for train [default: 15]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=4,
                    help='epochs to tolerate without performance increasing before early stopping')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-patience_step_scheduler', type=int, default=2,
                    help='number of steps to be patient during bad steps before reducing learning rate')

# model params
parser.add_argument('-model-type', type=str, default='shallow_cnn', help='architecture for training')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]') # goes to model config
parser.add_argument('-max-norm', type=float, default=0.0001, help='l2 constraint of parameters [default: 0.0001]')
parser.add_argument('-embed-type', type=str, default='fasttext.en.300d', help='pre-trained embeddings')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution') # goes to model config
parser.add_argument('-static', type=bool, default=False, help='fix the embedding')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-oof-dir', type=str, default='oof_predictions',
                    help='directory for saving out-of-fold predictions')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=bool, default=False, help='predict')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-output-name', type=str, help='filename for submission')
args = parser.parse_args()


def log_params(args):
    logging.info("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        logging.info("\t{}={}".format(attr.upper(), value))


def save_array(arr, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(file=os.path.join(path, name), arr=arr)


def cross_validation(args, master_df, test_df, text_field, label_fields):
    """
    Splits data into folds and performs cross validation and then saves
    out-of-fold predictions
    Arguments:
        args: command-line arguments for the experiment
        master_df: pd.DataFrame that contains entire training data
        test_df: pd.DataFrame that contains test data
        text_field: Field object for processing text
        label_fields: a list of Field objects each representing one category
    """

    # placeholders for oof predictions
    oof_train = np.zeros((len(master_df), len(CLASSES)), dtype=np.float32)
    oof_test = np.zeros((len(test_df), len(CLASSES), args.folds), dtype=np.float32)

    # build master and test dataset objects
    master_dataset = ToxicComments(df=master_df, text_field=text_field,
                                   label_fields=label_fields, examples=None)
    test_dataset = ToxicComments(df=test_df, text_field=text_field,
                                 label_fields=None, examples=None)

    # build vocab and embeddings
    text_field.build_vocab(master_dataset, min_freq=args.token_threshold,
                           vectors=args.embed_type)
    # update args and log them
    args.embed_num = len(text_field.vocab)
    args.embed_dim = text_field.vocab.vectors.size()[1]
    log_params(args)

    fold_size = len(master_dataset)//args.folds
    base_config = model_config[args.model_type]
    for fold_id in range(args.folds):
        # create train_examples, valid_examples, test_examples
        train_examples, valid_examples, test_examples = [], [], []
        fold_start = fold_size*fold_id
        fold_end = fold_start + fold_size
        train_examples.extend(master_dataset.examples[:fold_start])
        train_examples.extend(master_dataset.examples[fold_end:])
        valid_examples.extend(master_dataset.examples[fold_start:fold_end])
        test_examples.extend(test_dataset.examples)

        # build iterators for train, valid, test
        train_iter, valid_iter, test_iter = make_iterators(train_examples=train_examples,
                                                           valid_examples=valid_examples,
                                                           test_examples=test_examples,
                                                           text_field=text_field,
                                                           label_fields=label_fields)


        # load model or snapshot
        model = models.get_model(args=args, model_type=args.model_type,
                                 pretrained_vectors=text_field.vocab.vectors,
                                 base_config=base_config)
        if args.static:
            embs = next(model.parameters())
            embs.requires_grad = False

        if args.snapshot is not None:
            logging.info('\nLoading model from {}...'.format(args.snapshot))
            model.load_state_dict(torch.load(args.snapshot))

        # transfer model to GPU
        if args.cuda:
            torch.cuda.set_device(args.device)
            model = model.cuda()

        # training starts here
        logging.info('\n...............training starts.........................')
        train.train(train_iter, valid_iter, model, args)

        # load best model
        best_model_filename = sorted(glob.glob(os.path.join(args.save_dir, '*.pt')),
                                     key=lambda x: os.path.getmtime(x))[-1]
        model.load_state_dict(torch.load(best_model_filename))

        # predicting on valid fold and test
        valid_predictions = train.predict(valid_iter, model, args)
        oof_train[fold_start:fold_end] = valid_predictions
        oof_test[:, :, fold_id] = train.predict(test_iter, model, args)
        logging.info('\n...............training ends for fold{}................'.format(fold_id))

    # save oof-train predictions and test predictions
    oof_test = oof_test.mean(axis=2)
    save_array(oof_train, os.path.join(args.oof_path, args.exp_start_time), 'train.npy')
    create_submission(oof_test, test_df, args.exp_start_time + '.csv')


def main():
    # setup logging info
    if args.predict:
        exp_start_time = args.snapshot.split('/')[-2]
    else:
        exp_start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # add args
    args.exp_start_time = exp_start_time
    args.save_dir = os.path.join(args.save_dir, exp_start_time)
    args.cuda = torch.cuda.is_available()
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.class_num = len(CLASSES)

    # oof-dir for stacking
    oof_path = os.path.join(DATA_PATH, args.oof_dir)
    if not os.path.exists(oof_path):
        os.mkdir(oof_path)
    args.oof_path = oof_path

    logging_path = os.path.join(DATA_PATH, 'logs')
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)
    logging.basicConfig(filename=os.path.join(logging_path, exp_start_time + '.log'),
                        level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # load data
    logging.info('\nLoading data...')
    train_data = pd.read_csv(os.path.join(DATA_PATH, 'cleaned_data', 'cleaned_train.csv'))
    test_data = pd.read_csv(os.path.join(DATA_PATH, 'cleaned_data', 'cleaned_test.csv'))
    train_data[CLASSES] = train_data[CLASSES].astype(np.float32)

    # setup fields and make iterators
    text_field = data.Field(sequential=True, lower=True, eos_token='<pad>',
                            #tokenize=tokenizer,
                            fix_length=args.comment_size)
    label_fields = []
    for _ in CLASSES:
        label_fields.append(data.Field(sequential=False, lower=False, use_vocab=False,
                                       tensor_type=torch.FloatTensor))

    cross_validation(args, train_data, test_data, text_field, label_fields)


if __name__ == '__main__':
    main()
