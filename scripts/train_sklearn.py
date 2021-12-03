from scripts.dataloading import get_dataset, all_datasets
import argparse
from IPChecklists.Covers import KnapsackCover, SimpleCover
import random
import numpy as np
from pathlib import Path
import json
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', required = True)
parser.add_argument('--dataset', choices = all_datasets, required = True)
parser.add_argument('--binarization', choices = ['auto', 'expert', 'optbin', 'naive'], default = 'auto')
parser.add_argument('--model_type', choices = ['xgb', 'lr'], default = 'xgb')
parser.add_argument('--balance_train', action = 'store_true')
parser.add_argument('--balance_test', action = 'store_true')
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument('--fold', type = int, required = True, choices = list(range(6))) # 0 is using all
parser.add_argument('--seed', type = int, default = 42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

out_dir = Path(args.output_dir)
out_dir.mkdir(exist_ok = True, parents = True)

with open(out_dir/'args.json', 'w') as outfile:
    json.dump(vars(args), outfile)

dataset = get_dataset(args.dataset)(use_optbin = args.binarization == 'optbin', balance_train = args.balance_train,
                    balance_test = args.balance_test, use_naive = args.binarization == 'naive', fold = args.fold)

_, metrics = dataset.train_sklearn_model(model_type = args.model_type, dtype = 'binarized')

pickle.dump(
    metrics
, open(out_dir/'sklearn_metrics', 'wb'))

with (out_dir/'done').open('w') as f:
    f.write('done')    