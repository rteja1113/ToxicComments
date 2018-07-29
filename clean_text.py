import os
import re
import pandas as pd

DATA_PATH = 'data'


def remove_leaks(df):
    """
    Function for removing leaky parts in text
    like ips, urls, timestamps, usernames etc.
    :param df: DataFrame containing text
    :return:None
    """
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub('\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}', '', str(x)))
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub('http://.*com', '', str(x)))
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub('\d:\d\d\s{0,5}$', '', str(x)))
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub('\[\[User(.*)\|', '', str(x)))


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

    remove_leaks(train)
    remove_leaks(test)
    train = train.sample(frac=1).reset_index(drop=True)

    if not os.path.exists(os.path.join(DATA_PATH, 'cleaned_data')):
        os.mkdir(os.path.join(DATA_PATH, 'cleaned_data'))

    train.to_csv(os.path.join(DATA_PATH, 'cleaned_data', 'cleaned_train.csv'), index=None)
    test.to_csv(os.path.join(DATA_PATH, 'cleaned_data', 'cleaned_test.csv'), index=None)
