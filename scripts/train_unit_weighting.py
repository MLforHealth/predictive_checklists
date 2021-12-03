import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from IPChecklists.Constants import *
import json
import random
from scripts.dataloading import get_dataset, all_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from IPChecklists.lr_init import checklists_from_lr
from IPChecklists.utils import check_group_constraint, binary_metrics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', required = True)
parser.add_argument('--dataset', choices = all_datasets, required = True)
parser.add_argument('--binarization', choices = ['auto', 'expert', 'optbin', 'naive'], default = 'auto')
parser.add_argument('--C', type = float, required = True)
parser.add_argument('--balance_train', action = 'store_true')
parser.add_argument('--balance_test', action = 'store_true')
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument('--max_features', type = int)
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--fold', type = int, required = True, choices = list(range(6))) # 0 is using all
parser.add_argument('--use_basic', action = 'store_true', help = 'use a feature-group agnostic version of unit weighting')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

out_dir = Path(args.output_dir)
out_dir.mkdir(exist_ok = True, parents = True)

with open(out_dir/'args.json', 'w') as outfile:
    json.dump(vars(args), outfile)

dataset = get_dataset(args.dataset)(use_optbin = args.binarization == 'optbin', balance_train = args.balance_train,
                    balance_test = args.balance_test, use_naive = args.binarization == 'naive',
                    fold = args.fold)
df_train, df_test, train_ds, test_ds, all_cols = dataset.get_data()

train_X, train_Y = train_ds.binarized_df, df_train[train_ds.target_name]
test_X, test_Y = test_ds.binarized_df, df_test[test_ds.target_name]

pipe = Pipeline([
                 ('lr', LogisticRegression(penalty = 'l1', C = args.C, random_state = args.seed, solver = 'liblinear'))
                 ]).fit(train_X, train_Y)

checklists, train_metrics, test_metrics, N = checklists_from_lr(pipe, train_ds, test_ds, use_basic = args.use_basic)
if args.max_features is None or N <= args.max_features:
    if not args.use_basic:
        satisfies_group_constraint = True
    elif len(checklists) == 0:
        satisfies_group_constraint = False
    else: 
        satisfies_group_constraint = check_group_constraint(checklists[0], train_ds.feature_mapping)
    pickle.dump(checklists, open(out_dir/'checklists', 'wb'))
    pickle.dump({
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'satisfies_group_constraint': satisfies_group_constraint # same for all checklists
    }, open(out_dir/'metrics', 'wb'))

    # LR model stats
    pickle.dump({
        'train_metrics': binary_metrics(train_Y == train_ds.pos_label, pipe.predict(train_X)== train_ds.pos_label),
        'test_metrics': binary_metrics(test_Y== train_ds.pos_label, pipe.predict(test_X)== train_ds.pos_label),
        'n_features': (pipe.steps[0][1].coef_  != 0).sum()
    }, open(out_dir/'orig_model_stats', 'wb'))

with (out_dir/'done').open('w') as f:
    f.write('done')