import numpy as np
import logging
logging.basicConfig(level='WARNING')
from scripts import dataloading
import copy
from itertools import product

def combinations(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError(experiment)
    return globals()[experiment].hparams()


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError(experiment)
    return globals()[experiment].fname


#### write experiments here
'''
Experimental order:
- UnitWeightingBasic, UnitWeighting, SimpleCover, KnapsackCover, NoInit, NoInitFPR
- Sequential, SequentialFPR
- NoInitFPRFair
'''
all_datasets = dataloading.all_datasets
MAX_FEATURES_ALL = 8

class UnitWeightingBasic:
    fname = 'train_unit_weighting.py'

    @staticmethod
    def hparams():
        grid = {
        'dataset': all_datasets,
        'fold': list(range(0, 6)),
        'binarization': ['auto', 'optbin', 'naive'],
        'C': 10**np.linspace(-5, 1, 100),
        'balance_train': [True],
        'balance_test': [True],
        'max_features': [MAX_FEATURES_ALL],
        'use_basic': [True]
        }

        return combinations(grid)


class UnitWeighting:
    fname = 'train_unit_weighting.py'

    @staticmethod
    def hparams():
        grid = {
        'dataset': all_datasets,
        'fold': list(range(0, 6)),
        'binarization': ['auto', 'optbin', 'naive'],
        'C': 10**np.linspace(-5, 1, 100),
        'balance_train': [True],
        'balance_test': [True],
        'max_features': [MAX_FEATURES_ALL]
        }

        return combinations(grid)

class SimpleCover:
    fname = 'train_submodular.py'

    @staticmethod
    def hparams():
        all_grids = []
        for ds in all_datasets:
            max_N = dataloading.get_dataset(ds)(use_optbin = False, balance_train = False, balance_test = False).summarize()['n_orig_features']
            grid = {
                'dataset': [ds],
                'fold': list(range(0, 6)),
                'cover_type': ['simple'],
                'binarization': ['auto', 'optbin', 'naive'],
                'balance_train': [True],
                'balance_test': [True],
                'cover_label': [1, -1],
                'max_N': [min(MAX_FEATURES_ALL, max_N)]
            }
            grid['M'] = list(range(1, grid['max_N'][0] + 1))
            all_grids += combinations(grid)

        return all_grids

class KnapsackCover:
    fname = 'train_submodular.py'

    @staticmethod
    def hparams():
        all_grids = []
        for ds in all_datasets:
            max_N = dataloading.get_dataset(ds)(use_optbin = False, balance_train = False, balance_test = False).summarize()['n_orig_features']
            N_pos = dataloading.get_dataset(ds)(use_optbin = False, balance_train = True, balance_test = True).summarize()['n_positive']

            grid = {
                'dataset': [ds],
                'fold': list(range(0, 6)),
                'cover_type': ['knapsack'],
                'binarization': ['auto', 'optbin', 'naive'],
                'balance_train': [True],
                'balance_test': [True],
                'cover_label': [1, -1],
                'max_N': [min(MAX_FEATURES_ALL, max_N)],
                'knapsack_eps': [0.2, 0.5],
                'knapsack_budget': [N_pos*3., N_pos*2., float(N_pos), N_pos/1.5, N_pos/2, N_pos/3] # works for both labels because dataset is balanced
            }
            grid['M'] = list(range(1, grid['max_N'][0] + 1))
            all_grids += combinations(grid)

        return all_grids

class NoInit:
    fname = 'train_checklist.py'

    @staticmethod
    def hparams():
        all_grids = []
        for ds in all_datasets:
            max_N = dataloading.get_dataset(ds)(use_optbin = False, balance_train = False, balance_test = False).summarize()['n_orig_features']
            grid = {
            'training_mode': ['single'],
            'dataset': [ds],
            'fold': list(range(0, 6)),
            'binarization': ['auto', 'optbin', 'naive'],
            'solve_time': [60*60],
            'max_num_features': list(range(1, min(MAX_FEATURES_ALL + 1, max_N + 1))),
            'cost_func': ['01'],
            'balance_train': [True],
            'balance_test': [True]
            }
            all_grids += combinations(grid)

        return all_grids

class NoInitFPR:
    fname = 'train_checklist.py'

    @staticmethod
    def hparams():
        hparams = []
        for N in list(range(1, MAX_FEATURES_ALL + 1)):
            grid_ptsd = {
                'training_mode': ['single'],
                'dataset': ['PTSD'],
                'fold': list(range(0, 6)),
                'binarization': ['auto'],
                'solve_time': [60*60*4],
                'threads': [2],
                'max_num_features': [N],
                'M_constraint': list(range(1, min(N+1, 5))),
                'fnr_constraint': [0.05],
                'cost_func': ['FPR'],
                'balance_train': [False],
                'balance_test': [False],
                # 'allow_multiple_per_feature': [False, True]
                'allow_multiple_per_feature': [False]
            }

            grid_crrt = copy.deepcopy(grid_ptsd)
            grid_crrt['dataset'] = ['CRRT']
            grid_crrt['fnr_constraint'] = [0.20]

            hparams += combinations(grid_ptsd) + combinations(grid_crrt)

        return hparams

class Sequential:
    fname = 'train_checklist.py'

    @staticmethod
    def hparams():
        all_grids = []
        for ds in all_datasets:
            max_N = dataloading.get_dataset(ds)(use_optbin = False, balance_train = False, balance_test = False).summarize()['n_orig_features']
            grid = {
                'training_mode': ['sequential'],
                'dataset': [ds],
                'fold': list(range(0, 6)),
                'binarization': ['auto', 'optbin', 'naive'],
                'solve_time': [60*10],
                'max_num_features': [min(MAX_FEATURES_ALL, max_N)],
                'cost_func': ['01'],
                'balance_train': [True],
                'balance_test': [True],
                'use_pool': [True]
            }
            all_grids += combinations(grid)

        return all_grids

class SequentialFPR:
    fname = 'train_checklist.py'

    @staticmethod
    def hparams():
        grid_ptsd = {
            'training_mode': ['sequential'],
           'dataset': ['PTSD'],
           'fold': list(range(0, 6)),
           'binarization': ['auto'],
           'solve_time': [60*15],
           'max_num_features': [MAX_FEATURES_ALL],
           'fnr_constraint': [0.05],
           'cost_func': ['FPR'],
           'balance_train': [False],
            'balance_test': [False],
            'use_pool': [True],
            'threads': [4],
           #  'allow_multiple_per_feature': [False, True],
            'allow_multiple_per_feature': [False],
            # 'eval_attributes': [['sex','race', 'age_bin']]
        }

        grid_crrt = copy.deepcopy(grid_ptsd)
        grid_crrt['dataset'] = ['CRRT']
        grid_crrt['fnr_constraint'] = [0.20]
        # grid_crrt['eval_attributes'] = [['gender', 'ethnicity', 'age_bin']]

        return combinations(grid_ptsd)  + combinations(grid_crrt)

class NoInitFPRFair:
    fname = 'train_checklist.py'

    @staticmethod
    def hparams():
        hparams = []
        for N in list(range(1, MAX_FEATURES_ALL + 1)):
            grid_crrt = {
                'training_mode': ['single'],
                'fold': list(range(0, 6)),
                'dataset': ['CRRT'],
                'binarization': ['auto'],
                'solve_time': [60*60*4],
                'threads': [2],
                'max_num_features': [N],
                'M_constraint': list(range(1, min(N+1, 5))),
                'fnr_constraint': [0.20],
                'cost_func': ['FPR'],
                'balance_train': [False],
                'balance_test': [False],
                'use_pool': [True],
                'allow_multiple_per_feature': [False],
                'constraint_attr': ['ethnicity_gender'],
                'fnr_leq_limit': [0.20],
                'fpr_gap_limit': [0.15],
                'use_indicator': [True]
            }

           #  hparams += combinations(grid_ptsd)
            hparams += combinations(grid_crrt)
        
        return hparams

class Sklearn:
    fname = 'train_sklearn.py'

    @staticmethod
    def hparams():
        grid = {    
            'dataset': all_datasets,
            'model_type': ['xgb', 'lr'],
            'fold': list(range(0, 6)),
            'binarization': ['auto'],
            'balance_train': [True],
            'balance_test': [True],
        }

        return combinations(grid)
