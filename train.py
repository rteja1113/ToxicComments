import logging
import os
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_processing import CLASSES


def train(train_iter, dev_iter, model, args):
    """
    Performs training on train set and evaluation on validation set
    Arguments:
        train_iter: Iterator for train set
        dev_iter: Iterator for valid set
        model: PyTorch model
        args: command-line args
    """

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=args.patience_step_scheduler,
                                  factor=0.1, mode='max', verbose=True)
    best_auc = 0
    epoch_counter = 0
    last_best_epoch = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature = batch.text
            target_tensor = torch.cat([getattr(batch, C).data.view(-1, 1) for C in CLASSES], 1)
            target = torch.autograd.Variable(target_tensor)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            loss.backward()
            optimizer.step()

        epoch_counter += 1

        # evaluation
        dev_auc = eval(dev_iter, model, args)
        scheduler.step(dev_auc)

        # save weights if metric improves
        if dev_auc > best_auc:
            best_auc = dev_auc
            last_best_epoch = epoch_counter
            if args.save_best:
                save(model, args.save_dir, 'best', last_best_epoch)

        # early stopping criterion
        logging.info('\nend of epoch\n')
        logging.info('\nbest auc: {:.6f}\n'.format(best_auc))
        if epoch_counter - last_best_epoch >= args.early_stop:
            logging.info('early stop by {} epochs.'.format(args.early_stop))
            break
    return


def eval(data_iter, model, args):
    """
    evaluates on valid set
    Arguments:
        data_iter: Iterator for valid set
        model: PyTorch model
        args: command-line args
    returns:
        mean_auc: Mean of AUC for all classes
    """
    model.eval()
    avg_loss = 0
    predictions = torch.zeros((len(data_iter.dataset), len(CLASSES)))
    ground_truths = torch.zeros((len(data_iter.dataset), len(CLASSES)))
    example_index = 0
    for batch in data_iter:
        feature = batch.text
        target_tensor = torch.cat([getattr(batch, C).data.view(-1, 1) for C in CLASSES], 1)
        target = torch.autograd.Variable(target_tensor)
        if args.cuda:
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
    logging.info('\nEvaluation - loss: {:.6f}  mean_auc: {:.6f}\n'.format(avg_loss,
                                                                   mean_auc))
    return mean_auc


def predict(data_iter, model, args):
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
    for batch in data_iter:
        feature = batch.text
        if args.cuda:
            feature = feature.cuda()

        logit = model(feature)
        batch_preds = F.sigmoid(logit)
        predictions[example_index:(example_index + feature.size()[0])] = batch_preds.data.cpu()
        example_index += feature.size()[0]

    return predictions.numpy()


def save(model, save_dir, save_prefix, steps):
    """
    Serializes the model
    Arguments:
        model: PyTorch model
        save_dir: directory in which model is serialized
        save_prefix: prefix for file
        steps: training step at which serialization is done
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)