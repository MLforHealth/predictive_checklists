import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from IPChecklists.dataset import BinaryDataset
from IPChecklists.model_cplex import ChecklistMIP, Checklist
from sklearn.metrics import accuracy_score, confusion_matrix
from IPChecklists.Constants import *
from IPChecklists.constraints_cplex import FNRConstraint, MaxNumFeatureConstraint, MConstraint, GroupFPRGapConstraint, GroupFNRConstraint
import json
import random
from scripts.dataloading import get_dataset, all_datasets
from IPChecklists.SolutionPool import SolutionPool

POOL_NAMES = ['SimpleCover', 'KnapsackCover', 'UnitWeighting', 'NoInit', 'NoInitFPR']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', required = True)
parser.add_argument('--training_mode', choices = ['sequential', 'single'], required = True)
parser.add_argument('--dataset', choices = all_datasets, required = True)
parser.add_argument('--binarization', choices = ['auto', 'expert', 'optbin', 'naive'], default = 'auto')
parser.add_argument('--solve_time', type = int, default = 3600)
parser.add_argument('--cost_func', choices = ['01', 'FPR'], required = True)
parser.add_argument('--max_num_features', type = int, default = 0, help = "constraint on # features, set 0 to be unlimited")
parser.add_argument('--balance_train', action = 'store_true')
parser.add_argument('--balance_test', action = 'store_true')
parser.add_argument('--M_constraint', type = int, default = 0, help = "constraint on # positive features, set 0 to be unlimited")
parser.add_argument('--num_features_op', type = str, choices = ['==','<='], default = '<=')
parser.add_argument('--fnr_constraint', type = float, required = False)
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--threads', type = int, default = 1)
parser.add_argument('--allow_multiple_per_feature', help = 'allow multiple checklist items for each feature', action = 'store_true')
parser.add_argument('--save_logs', action = 'store_true')
parser.add_argument('--use_indicator', action = 'store_true', help = 'use indicator constraints instead of big-M in formulation')
parser.add_argument('--fold', type = int, required = True, choices = list(range(6))) # 0 is using all

# fairness constraints
parser.add_argument('--constraint_attr', type = str)
parser.add_argument('--fnr_leq_limit', type = float, default = 0.05)
parser.add_argument('--fpr_gap_limit', type = float, default = 0.10)

# pool of solutions
parser.add_argument('--use_pool', action = 'store_true')
parser.add_argument('--pool_dir', type = str, default = None, help = 'defaults to output_dir.parent.parent')
parser.add_argument('--pool_subset', type = str, choices = POOL_NAMES + ['All'], nargs = '+', default = ['All'])
args = parser.parse_args()

random.seed(args.seed)

out_dir = Path(args.output_dir)
out_dir.mkdir(exist_ok = True, parents = True)

with open(out_dir/'args.json', 'w') as outfile:
    json.dump(vars(args), outfile)

print(json.dumps(vars(args), indent = 4, sort_keys = True))

dataset = get_dataset(args.dataset)(use_optbin = args.binarization == 'optbin', balance_train = args.balance_train,
                    balance_test = args.balance_test, use_naive = args.binarization == 'naive', fold = args.fold)
df_train, df_test, train_ds, test_ds, all_cols = dataset.get_data()

N_max = args.max_num_features if args.max_num_features > 0 else train_ds.binarized_df.shape[1]
N_min = 0 if args.num_features_op == '<=' else N_max
if args.M_constraint > 0:
    M_min, M_max = 0, args.M_constraint
else:
    M_min, M_max = 0, N_max

pool = SolutionPool(args.dataset, args.binarization)
usable_pools = POOL_NAMES if 'All' in args.pool_subset else args.pool_subset
if args.use_pool:
    pool_dir = Path(args.pool_dir) if args.pool_dir is not None else out_dir.parent.parent
    for dir in pool_dir.glob('*/'):
        if dir.name in POOL_NAMES and dir.name in usable_pools:
            pool.add_from_dir(dir, enforce_fold = args.fold)
    print("Added %s initial solutions to pool" % len(pool))
    print(pool.df['origin'].value_counts())

if args.training_mode == 'single':
    model = ChecklistMIP(train_ds, model_name = args.dataset, cost_func = args.cost_func, one_feature_per_group=not args.allow_multiple_per_feature,
        compress = (args.constraint_attr is None))
    N_constraint = MaxNumFeatureConstraint(args.num_features_op, args.max_num_features) if args.max_num_features > 0 else None
    M_constraint = MConstraint('<=' , args.M_constraint) if args.M_constraint > 0 else None
    if args.cost_func == 'FPR':
        model.add_constraint(FNRConstraint(args.fnr_constraint))

    if args.constraint_attr:
        model.add_constraint(GroupFNRConstraint(args.constraint_attr, args.fnr_leq_limit))
        model.add_constraint(GroupFPRGapConstraint(args.constraint_attr, args.fpr_gap_limit))

    model.build_problem(N_constraint = N_constraint, M_constraint = M_constraint, use_indicator = args.use_indicator)

    if args.use_pool:
        subset = pool.get_subset(N = N_max, M = M_max)
        for s in subset:
            model.add_initial_solution_from_checklist(s)

    model.solve(max_seconds = args.solve_time, display_progress = True, seed = args.seed, threads = args.threads)

    results = model.get_solution_info()

    if model.solved:
        checklist = model.to_checklist()
        results['train_results'] = checklist.get_metrics(train_ds)
        results['test_results'] = checklist.get_metrics(test_ds)

        if args.save_logs:
            model.get_progress_logs().to_csv(out_dir/'logs.csv')
        pickle.dump(checklist, (out_dir/'checklist').open('wb'))

    pickle.dump(results, (out_dir/'metrics').open('wb'))

else:
    checklists, metas, train_metrics, test_metrics = [], [], [], []
    for N in range(1, N_max + 1):
        for M in range(1, N+1):
            model = ChecklistMIP(train_ds, model_name = args.dataset, cost_func = args.cost_func, one_feature_per_group=not args.allow_multiple_per_feature,
                        compress = (args.constraint_attr is None))
            N_constraint = MaxNumFeatureConstraint('<=', N)
            M_constraint = MConstraint('<=' , M)
            if args.cost_func == 'FPR':
                model.add_constraint(FNRConstraint(args.fnr_constraint))

            if args.constraint_attr:
                model.add_constraint(GroupFNRConstraint(args.constraint_attr, args.fnr_leq_limit))
                model.add_constraint(GroupFPRGapConstraint(args.constraint_attr, args.fpr_gap_limit))

            model.build_problem(N_constraint = N_constraint, M_constraint = M_constraint, use_indicator = args.use_indicator)

            if args.use_pool:
                subset = pool.get_subset(N = N, M = M)
                for s in subset:
                    model.add_initial_solution_from_checklist(s)

            model.solve(max_seconds = args.solve_time, display_progress = True, seed = args.seed, threads = args.threads)

            results = model.get_solution_info()

            if model.solved:
                checklist = model.to_checklist()
                if args.use_pool:
                    pool.add_to_pool(checklist, 'Sequential')

                train_metrics.append(checklist.get_metrics(train_ds))
                test_metrics.append(checklist.get_metrics(test_ds))

                results = {
                    **results,
                    **{
                        'N_cons': N,
                        'M_cons': M,
                        'N': checklist.N,
                        "M": checklist.M,
                        'num_starts': len(subset) if args.use_pool else 0
                    }
                }
                metas.append(results)
                checklists.append(checklist)

    pickle.dump(checklists, open(out_dir/'checklists', 'wb'))
    pickle.dump(metas, open(out_dir/'metas', 'wb'))
    pickle.dump({
    'train_metrics': train_metrics,
    'test_metrics': test_metrics
    }, open(out_dir/'metrics', 'wb'))

with (out_dir/'done').open('w') as f:
    f.write('done')
