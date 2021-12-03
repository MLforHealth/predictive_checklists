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
parser.add_argument('--cover_type', choices = ['simple', 'knapsack'])
parser.add_argument('--cover_label', type = int, choices = [1, -1])
parser.add_argument('--max_N', type = int, required = True)
parser.add_argument('--M', type = int, required = True)
parser.add_argument('--knapsack_budget', type = float, default = 10000)
parser.add_argument('--knapsack_eps', type = float, default = 0.5)
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
df_train, df_test, train_ds, test_ds, all_cols = dataset.get_data()

if args.cover_type == 'knapsack':
    cover = KnapsackCover(train_ds, M = args.M, max_N = args.max_N, budget = args.knapsack_budget, eps = args.knapsack_eps,
                enforce_group_constraint = True) 
elif args.cover_type == 'simple':
    cover = SimpleCover(train_ds, M = args.M, max_N = args.max_N, enforce_group_constraint = True) 

cover.solve()    
checklists = [i.to_checklist() for i in cover.get_results(return_intermediate = True)]
pickle.dump(checklists, open(out_dir/'checklists', 'wb'))

train_metrics = [check.get_metrics(train_ds) for check in checklists]
test_metrics = [check.get_metrics(test_ds) for check in checklists]

pickle.dump({
    'train_metrics': train_metrics,
    'test_metrics': test_metrics
}, open(out_dir/'metrics', 'wb'))

with (out_dir/'done').open('w') as f:
    f.write('done')    