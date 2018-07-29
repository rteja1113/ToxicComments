import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from data_processing import DATA_PATH, CLASSES
from data_processing import create_submission
from models import Stacker
from train import save


test_df = pd.read_csv(os.path.join(DATA_PATH, 'cleaned_data', 'cleaned_test.csv'))
EPOCHS = 18


class TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        if self.target_tensor is not None:
            return self.data_tensor[index], self.target_tensor[index]
        else:
            return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def make_loader(input, target):
    """
    Returns a DataLoader for input, target
    Arguments:
          input: input data
          target: targer data
    Returns:
          torch DataLoader
    """
    stack_dataset = TensorDataset(input, target)
    stack_dataloader = DataLoader(stack_dataset, batch_size=64)
    return stack_dataloader


def train(train_iter, dev_iter, model):
    """
    Performs training on train set and evaluation on validation set
    Arguments:
        train_iter: Iterator for train set
        dev_iter: Iterator for valid set
        model: PyTorch model
        args: command-line args
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=1,
                                  factor=0.1, mode='max', verbose=True)
    best_auc = 0
    epoch = 0
    last_epoch = 0
    model.train()
    for epoch in range(1, EPOCHS+1):
        for feature, target in train_iter:
            feature = torch.autograd.Variable(feature)
            target = torch.autograd.Variable(target)
            feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            loss.backward()
            optimizer.step()

        dev_auc = eval(dev_iter, model)
        scheduler.step(dev_auc)
        if dev_auc > best_auc:
            best_auc = dev_auc
            last_epoch = epoch
            save(model, DATA_PATH, 'stack', 'model')


def eval(data_iter, model):
    """
    evaluates on valid set
    Arguments:
        data_iter: Iterator for valid set
        model: PyTorch model
    Returns:
        mean_auc: Mean of AUC for all classes
    """
    model.eval()
    avg_loss = 0
    predictions = torch.zeros((len(data_iter.dataset), len(CLASSES)))
    ground_truths = torch.zeros((len(data_iter.dataset), len(CLASSES)))
    example_index = 0
    for feature, target in data_iter:
        feature = torch.autograd.Variable(feature)
        target = torch.autograd.Variable(target)
        feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.binary_cross_entropy_with_logits(logit, target)
        batch_preds = F.sigmoid(logit)
        predictions[example_index:(example_index + feature.size()[0])] = batch_preds.data.cpu()
        ground_truths[example_index:(example_index + feature.size()[0])] = target.data.cpu()
        example_index += feature.size()[0]
        avg_loss += loss.data[0]

    # gather AUCs for all classes
    auc_list = [roc_auc_score(ground_truths.numpy()[:, i], predictions.numpy()[:, i])
                for i in range(len(CLASSES))]
    mean_auc = sum(auc_list) / len(auc_list)
    size = len(data_iter.dataset)
    avg_loss /= size
    print('\nEvaluation - loss: {:.6f}  mean_auc: {:.6f}\n'.format(avg_loss, mean_auc))
    return mean_auc


def predict(data_iter, model):
    """
    feed-forwards on data_iter and return predictions
    Arguments:
        data_iter: Iterator for predicting
        model: PyTorch model
        args: command-line args
    returns:
        np.ndarray of predictions
    """
    model.eval()
    predictions = torch.zeros((len(data_iter.dataset), len(CLASSES)))
    example_index = 0
    for feature in data_iter:
        feature = torch.autograd.Variable(feature)
        feature = feature.cuda()
        logit = model(feature)
        batch_preds = F.sigmoid(logit)
        predictions[example_index:(example_index + feature.size()[0])] = batch_preds.data.cpu()
        example_index += feature.size()[0]

    return predictions.numpy()


def cross_validation(master, target, test):
    """
    Performs cross-validation on oof-data
    Arguments:
        master: oof-predictions of entire train set
        target: target data of entire train set
        test: test set

    """
    kf = KFold(n_splits=5, shuffle=True, random_state=2018)
    split_gen = kf.split(master)
    oof_train = np.zeros((len(master), len(CLASSES)), dtype=np.float32)
    oof_test = np.zeros((len(test), len(CLASSES), 5), dtype=np.float32)

    # begin training
    for fold, (train_index, valid_index) in enumerate(split_gen):
        # create train_data, valid_data and build iterators for train, valid, test
        train_index = torch.from_numpy(train_index)
        valid_index = torch.from_numpy(valid_index)
        train_x = master[train_index]
        train_y = target[train_index]
        valid_x = master[valid_index]
        valid_y = target[valid_index]

        # make DataLoaders for train, valid, test
        input_size = train_x.size()[1]
        train_loader = make_loader(train_x, train_y)
        valid_loader = make_loader(valid_x, valid_y)
        test_loader = make_loader(test, None)

        # model
        stack_model = Stacker(input_size)
        stack_model = stack_model.cuda()

        # training
        train(train_loader, valid_loader, stack_model)

        # predicting on test set
        best_model_filename = os.path.join(DATA_PATH, 'stack_steps_model.pt')
        stack_model.load_state_dict(torch.load(best_model_filename))
        oof_test[:, :, fold] = predict(test_loader, stack_model)
        print('End of fold-{}'.format(fold))

    oof_test = oof_test.mean(axis=2)
    #save_array(oof_train, os.path.join(args.oof_path, args.exp_start_time), 'train.npy')
    create_submission(oof_test, test_df, 'stack_level0' + '.csv')


def check_order(train_paths, test_paths):
    """
    Checks order of file names for stacking purpose
    Arguments:
         train_paths: list of train paths containing oof train predictions
         test_paths: list of test paths containing oof test predictions
    """
    train_names = [p.split('/')[-2] for p in train_paths]
    test_names = [p.split('/')[-1].split('.')[0] for p in test_paths]
    for train_name, test_name in zip(train_names, test_names):
        assert train_name == test_name


def main():
    # obtain paths
    oof_train_paths = glob.glob(os.path.join(DATA_PATH, 'oof_predictions',
                                             '*', 'train.npy'))
    oof_train_paths = sorted(oof_train_paths, key=lambda x: os.path.getmtime(x))
    test_paths = glob.glob(os.path.join(DATA_PATH,
                                        'submissions', '*.csv'))
    test_paths = sorted(test_paths, key=lambda x: os.path.getmtime(x))

    # some preliminary checks
    assert len(oof_train_paths) == len(test_paths)
    check_order(oof_train_paths, test_paths)

    # load train and test inputs
    train_x = [np.load(f) for f in oof_train_paths]
    train_x = np.concatenate(train_x, axis=1)
    test_x = [pd.read_csv(f, usecols=CLASSES).values for f in test_paths]
    test_x = np.concatenate(test_x, axis=1).astype(np.float32)

    # load target
    train_y = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'), usecols=CLASSES).values
    train_y = train_y.astype(np.float32)
    # convert to torch tensors
    train_x = torch.from_numpy(train_x)
    test_x = torch.from_numpy(test_x)
    train_y = torch.from_numpy(train_y)
    cross_validation(train_x, train_y, test_x)


if __name__ == '__main__':
    main()
