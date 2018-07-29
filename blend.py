import os
import glob
import pandas as pd
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
DATA_PATH = 'data'

if __name__ == '__main__':
    lgb_sub = pd.read_csv(os.path.join(DATA_PATH, 'independent_stack.csv'))
    xgb_sub = pd.read_csv(os.path.join(DATA_PATH, 'independent_stack_xgb.csv'))
    hap_sub = pd.read_csv(os.path.join(DATA_PATH, 'hapness_submission.csv'))
    sub = pd.DataFrame()
    sub['id'] = lgb_sub['id']
    for c in CLASSES:
        sub[c] = 0.33*lgb_sub[c] + 0.33*xgb_sub[c] + 0.33*hap_sub[c]

    sub.to_csv(os.path.join(DATA_PATH, 'blend_independent_stack.csv'), index=None)