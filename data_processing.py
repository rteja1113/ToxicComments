import re
import logging
import os
import time
import random
import numpy as np
import pandas as pd
import spacy
from torchtext import data

random.seed(2018)

DATA_PATH = 'data'
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
spacy_en = spacy.load('en')
filter_atts = ['is_space', 'is_bracket', 'is_quote', 'like_url', 'like_email',
               'is_stop']

class ToxicComments(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, df, text_field, label_fields, examples=None, **kwargs):
        """Create an dataset instance given a DataFrame and fields.

        Arguments:
            df: DataFrame containing text and labels
            text_field: The field that will be used for text data.
            label_fields: list of fields that will be used for label data.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        fields = [('text', text_field)]
        if label_fields:
            fields.extend([(CLASSES[i], label) for i, label in enumerate(label_fields)])
        start_time = time.time()
        if examples is None:
            examples = []
            for i, row in df.iterrows():
                sample = [row['comment_text']]
                if label_fields:
                    sample_y = [row[label] for label in CLASSES]
                    sample.extend(sample_y)
                examples.extend([data.Example.fromlist(sample, fields)])

        logging.info('\ntime taken to prepare examples: {}'.format(time.time() - start_time))
        super(ToxicComments, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, df, text_field, label_fields, examples=None, **kwargs):
        """Create dataset objects for splits of the ToxicComments dataset.

        Arguments:
            df: DataFrame containing text and labels
            text_field: The field that will be used for the sentence.
            label_fields: list of fields that will be used for label data.
        """
        if isinstance(df, pd.DataFrame):
            examples = cls(df, text_field, label_fields, **kwargs).examples
        return cls(df=None, text_field=text_field, label_fields=label_fields,
                   examples=examples)


def get_conversion(field):
    """
    computes conversion ratio
    defined as
    (# of words found in pre-trained embeddings)/(# of words in vocab)
    Arguments:
        field: Field object for text field
    returns:
        conversion ratio
    """
    vsum = field.vocab.vectors.sum(1)
    return (vsum > 0).sum()/len(vsum)


def make_iterators(train_examples, valid_examples, test_examples,
                   text_field, label_fields, **kwargs):
    """
    makes iterators for train_df, valid_df and test_df
    builds vocab object using train data, valid data and test data

    Arguments:
        train_df: DataFrame of train set
        valid_df: DataFrame of valid set
        test_df: DataFrame of test set
        text_field: Field object for processing text
        label_fields: list of Field objects, one for each class
        cache: list of cache file names containing examples
               [train_cache, valid_cache, test_cache]
        args: command-line args
    returns:
        Iterators for train, valid and test
    """

    train_data = ToxicComments.splits(df=None, text_field=text_field,
                                      label_fields=label_fields,
                                      examples=train_examples, **kwargs)
    valid_data = ToxicComments.splits(df=valid_examples, text_field=text_field,
                                      label_fields=label_fields,
                                      examples=valid_examples, **kwargs)
    test_data = ToxicComments.splits(df=test_examples, text_field=text_field,
                                     label_fields=None,
                                     examples=test_examples, **kwargs)

    train_iter, valid_iter, test_iter = data.Iterator.splits(
                                        (train_data, valid_data, test_data),
                                        batch_sizes=(128, 128, 128), sort=False,
                                        repeat=False, **kwargs)
    return train_iter, valid_iter, test_iter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r":", " ", string)
    string = re.sub(r"=", " ", string)
    string = string.lower()
    return string.strip()


def binary_search(a_list, item):
    """Performs iterative binary search to find the position of an integer in a given, sorted, list.
    a_list -- sorted list of integers
    item -- integer you are searching for the position of
    """

    first = 0
    last = len(a_list) - 1

    while first <= last:
        i = int((first + last) / 2)

        if a_list[i] == item:
            return True
        elif a_list[i] > item:
            last = i - 1
        elif a_list[i] < item:
            first = i + 1
        else:
            return False


def is_bad_word(text):
    hit = False
    for word in common_bad_words:
        if word in text:
            hit = True
    return hit


def tokenizer(text):
    filtered_text = []
    text = clean_str(text)
    for tok in spacy_en.tokenizer(text):
        init_filter = None
        for attr in filter_atts:
            if init_filter is None:
                init_filter = not getattr(tok, attr)
            else:
                init_filter = init_filter and (not getattr(tok, attr))
        #init_filter = init_filter and getattr(tok, 'is_ascii')
        if init_filter:
            #if binary_search(reference_itos, tok.text):
            #    filtered_text.append(tok.text)
            #elif is_bad_word(tok.text):
                #for word in common_bad_words:
                    #if word in tok.text:
                    #    filtered_text.append(word)
            #else:
            filtered_text.append(tok.text)

    return filtered_text


#def tokenizer(text):
#    return list(text)


def create_submission(predictions, df, output_name):
    """
    saves predictions according to sample submission
    predictions: np.ndarray of predictions
    df: test DataFrame or sample_submission DataFrame
    output_name: output name
    """
    submission = pd.DataFrame()
    submission['id'] = df['id']
    for i, c in enumerate(CLASSES):
        submission[c] = predictions[:, i]
    submission.to_csv(os.path.join(DATA_PATH, output_name), index=None)


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(DATA_PATH, 'cleaned_data', 'cleaned_train.csv'))
    test = pd.read_csv(os.path.join(DATA_PATH, 'cleaned_data', 'cleaned_test.csv'))
    train[CLASSES] = train[CLASSES].astype(np.int32)
    text_field = data.Field(sequential=True, lower=True, eos_token='<pad>',
                            tokenize=tokenizer)
    label_fields = [data.Field(sequential=False, lower=False, use_vocab=False) for _ in CLASSES]
    train_iter, dev_iter, test_iter = make_iterators(df=train, text_field=text_field,
                                                     label_fields=label_fields)

    ############################### Check Iterarors ###########################################
    print()
    for batch in train_iter:
        feature = batch.text
        break
    for batch in test_iter:
        feature = batch.text