import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
import random
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score
from cplex.callbacks import IncumbentCallback, HeuristicCallback, MIPInfoCallback
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold

def infer_types(df, subset = None):
    '''
    input: df[subset], a dataframe of features
    output: a dictionary {col: type}, where type in ['categorical', 'ordinal', numeric']
    '''
    temp = df if subset is None else df[subset]
    res = {}
    for i in temp:
        if is_string_dtype(temp[i]) or len(temp[i].unique()) == 2:
            res[i] = 'categorical'
        elif temp[i].dtype in (np.int32, np.int64, int) and len(temp[i].unique()) < 50:
            res[i] = 'ordinal'
        else:
            res[i] = 'numeric'
    return res

def random_supersample(df, target_name):
    rs = RandomOverSampler(sampling_strategy='minority', random_state = 42)
    return df.loc[rs.fit_resample(df.index.values.reshape(-1, 1), df[target_name] == df[target_name].iloc[0])[0].squeeze()]

def capsCase(st):
    temp = st.split(' ')
    return ' '.join([i[0].upper() + i[1:] for i in temp])

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)

def set_seed_wrapper(f, seed = 42):
    def mod_f(*args, **kwargs):
        st0 = np.random.get_state()
        st1 = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        ans = f(*args, **kwargs)
        np.random.set_state(st0)
        random.setstate(st1)
        return ans

    return mod_f

def split_folds(df, target_name, fold, random_seed = 42):
    skf = StratifiedKFold(n_splits = 5, random_state = random_seed, shuffle = True)
    df['fold_id'] = None
    for c, (_, fold_idx) in enumerate(skf.split(list(range(df.shape[0])), df[target_name])):
        df.iloc[fold_idx, df.columns.get_loc('fold_id')] = c + 1

    if fold == 0:
        return (df.drop(columns = ['fold_id']), df.drop(columns = ['fold_id']))
    else:
        return (df[df.fold_id != fold].drop(columns = ['fold_id']), df[df.fold_id == fold].drop(columns = ['fold_id']))
    

def CIs(row):
    return {
        'mean': row.mean(),
        'lower': np.quantile(row, 0.025),
        'upper': np.quantile(row, 0.975)
    }

def bin_var(bins, x):
    for c, i in enumerate(bins):
        if (x >= i[0] and x < i[1]) or (c == len(bins) -1 and x == i[1]):
            return f'{i[0]}-{i[1]}'

def binary_metrics(targets, preds, return_arrays = False):    
    if len(targets) == 0:
        return {}
    res = {'accuracy': accuracy_score(targets, preds)}
    CM = confusion_matrix(targets, preds, labels = [0, 1])

    res['n_samples'] = len(targets)

    res['TN'] = CM[0][0].item()
    res['FN'] = CM[1][0].item()
    res['TP'] = CM[1][1].item()
    res['FP'] = CM[0][1].item()

    res['error'] = res['FN'] + res['FP']

    if res['TP']+res['FN'] == 0:
        res['TPR'] = 0
        res['FNR'] = 1
    else:
        res['TPR'] = res['TP']/(res['TP']+res['FN'])
        res['FNR'] = res['FN']/(res['TP']+res['FN'])

    if res['FP']+res['TN'] == 0:
        res['FPR'] = 1
        res['TNR'] = 0
    else:
        res['FPR'] = res['FP']/(res['FP']+res['TN'])
        res['TNR'] = res['TN']/(res['FP']+res['TN'])

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds
    res['precision'] = res['TP'] / (res['TP'] + res['FP']) if (res['TP'] + res['FP']) > 0 else 0
    res['pred_prevalence'] = (res['TP'] + res['FP'])/res['n_samples']    
    res['prevalence'] = (res['TP'] + res['FN'])/res['n_samples']  
    res['balanced_acc'] = balanced_accuracy_score(targets, preds)
    
    return res

def probability_metrics(targets, preds):    
    return{
        'n_samples': len(targets),
        'AUROC': roc_auc_score(targets, preds),
        'AUPRC': average_precision_score(targets, preds)
    }    

def check_group_constraint(checklist, feature_mapping):
    selected_cols = set(checklist.get_selected_cols())
    for i in feature_mapping:
        if len(selected_cols.intersection(set(feature_mapping[i].binarized_names))) > 1:
            return False    
    return True            

def convert_target_domain(target):
    if np.all(np.sort(np.unique(target)) == np.array([-1, +1])):
        return (target + 1)/2
    else:
        return target

def per_group_metrics(ds, preds, attributes):
    assert(all([i in ds.protected_attrs for i in attributes]))
    rows = {}
    for attr in attributes:   
        for grp in ds.protected_groups[attr].unique():
            mask = ds.protected_groups[attr] == grp
            preds_i = preds[mask]
            targets_i = convert_target_domain(ds.target[mask])
            metrics_i = binary_metrics(targets_i, preds_i)
            rows[grp] = metrics_i
    return pd.DataFrame(rows)

def get_max_disparity(df, grps, metrics = ['FPR', 'FNR'], prefix = 'training'):
    '''
    Given a dataframe output from per_group_metrics, returns the max disparity and largest value across grps for metrics
    '''
    res = {}
    for met in metrics:
        row = df.loc[met, grps]
        res[prefix + '_' + met + ' (Maximum Disparity)'] = np.max(row) - np.min(row) # largest gap
        res[prefix + '_' + met + ' (Largest)'] = np.max(row) # worst value
    return pd.Series(res)

class StatsCallback(MIPInfoCallback):

    def initialize(self, store_solutions = False, solution_start_idx = None, solution_end_idx = None):

        # scalars
        self.times_called = 0
        self.start_time = None

        # stats that are stored at every call len(stat) = times_called
        self.runtimes = []
        self.nodes_processed = []
        self.nodes_remaining = []
        self.lowerbounds = []

        # stats that are stored at every incumbent update
        self.best_objval = float('inf')
        self.update_iterations = []
        self.incumbents = []
        self.upperbounds = []

        self.store_incumbent_solutions = store_solutions

        if self.store_incumbent_solutions:
            assert solution_start_idx <= solution_end_idx
            self.start_idx, self.end_idx = int(solution_start_idx), int(solution_end_idx)
            self.process_incumbent = self.record_objval_and_solution_before_incumbent
        else:
            self.process_incumbent = self.record_objval_before_incumbent

    def __call__(self):
        self.times_called += 1
        if self.start_time is None:
            self.start_time = self.get_start_time()
        self.runtimes.append(self.get_time())
        self.lowerbounds.append(self.get_best_objective_value())
        self.nodes_processed.append(self.get_num_nodes())
        self.nodes_remaining.append(self.get_num_remaining_nodes())
        self.process_incumbent()

    def record_objval_before_incumbent(self):
        if self.has_incumbent():
            self.record_objval()
            self.process_incumbent = self.record_objval

    def record_objval_and_solution_before_incumbent(self):
        if self.has_incumbent():
            self.record_objval_and_solution()
            self.process_incumbent = self.record_objval_and_solution

    def record_objval(self):
        objval = self.get_incumbent_objective_value()
        if objval < self.best_objval:
            self.best_objval = objval
            self.update_iterations.append(self.times_called)
            self.upperbounds.append(objval)

    def record_objval_and_solution(self):
        objval = self.get_incumbent_objective_value()
        if objval < self.best_objval:
            self.update_iterations.append(self.times_called)
            self.incumbents.append(self.get_incumbent_values(self.start_idx, self.end_idx))
            self.upperbounds.append(objval)
            self.best_objval = objval

    def check_stats(self):
        """checks stats rep at any point during the solution process"""

        # try:
        n_calls = len(self.runtimes)
        n_updates = len(self.upperbounds)
        assert n_updates <= n_calls

        if n_calls > 0:

            assert len(self.nodes_processed) == n_calls
            assert len(self.nodes_remaining) == n_calls
            assert len(self.lowerbounds) == n_calls

            lowerbounds = np.array(self.lowerbounds)
            for ub in self.upperbounds:
                assert np.all(ub + 1e-5 >= lowerbounds)

            runtimes = np.array(self.runtimes) - self.start_time
            nodes_processed = np.array(self.nodes_processed)

            is_increasing = lambda x: (np.diff(x) >= -1e-5).all()
            assert is_increasing(runtimes)
            assert is_increasing(nodes_processed)
            assert is_increasing(lowerbounds)

        if n_updates > 0:

            assert len(self.update_iterations) == n_updates
            if self.store_incumbent_solutions:
                assert len(self.incumbents) == n_updates

            update_iterations = np.array(self.update_iterations)
            upperbounds = np.array(self.upperbounds)
            gaps = (upperbounds - lowerbounds[update_iterations - 1]) / (np.finfo(np.float).eps + upperbounds)

            is_increasing = lambda x: (np.diff(x) >= -1e-5).all()
            assert is_increasing(update_iterations)
            assert is_increasing(-upperbounds)
            assert is_increasing(-gaps)

        return True

    def get_stats(self, return_solutions = False):

        assert self.check_stats()
        import pandas as pd
        MAX_UPPERBOUND = float('inf')
        MAX_GAP = 1.00

        stats = pd.DataFrame({
            'runtime': [t - self.start_time for t in self.runtimes],
            'nodes_processed': list(self.nodes_processed),
            'nodes_remaining': list(self.nodes_remaining),
            'lowerbound': list(self.lowerbounds)
            })

        upperbounds = list(self.upperbounds)
        update_iterations = list(self.update_iterations)
        incumbents = [] #empty placeholder            
            
        if len(update_iterations) == 0:
            return stats, incumbents
            
        # add upper bounds as well as iterations where the incumbent changes

        if update_iterations[0] > 1:
            update_iterations.insert(0, 1)
            upperbounds.insert(0, MAX_UPPERBOUND)
        row_idx = [i - 1 for i in update_iterations]

        stats = stats.assign(iterations = pd.Series(data = update_iterations, index = row_idx),
                             upperbound = pd.Series(data = upperbounds, index = row_idx))
        stats['incumbent_update'] = np.where(~np.isnan(stats['iterations']), True, False)
        stats = stats.fillna(method = 'ffill')

        # add relative gap
        gap = (stats['upperbound'] - stats['lowerbound']) / (stats['upperbound'] + np.finfo(np.float).eps)
        stats['gap'] = np.fmin(MAX_GAP, gap)

        # add model ids
        if return_solutions:
            incumbents = list(self.incumbents)
            model_ids = range(len(incumbents))
            row_idx = [i - 1 for i in self.update_iterations]
            stats = stats.assign(model_ids = pd.Series(data = model_ids, index = row_idx))
            stats = stats[['runtime',
                           'gap',
                           'upperbound',
                           'lowerbound',
                           'nodes_processed',
                           'nodes_remaining',
                           'model_id',
                           'incumbent_update']]

        else:
            stats = stats[['runtime',
                           'gap',
                           'upperbound',
                           'lowerbound',
                           'nodes_processed',
                           'nodes_remaining']]

        return stats, incumbents

class ProcessedDataset():
    def __init__(self, mat, target, I_plus, I_minus, n):
        self.mat = mat
        self.target = target
        self.n = n
        self.I_plus = I_plus
        self.I_minus = I_minus
        self.n_samples_plus = self.n[self.I_plus].sum()
        self.n_samples_minus = self.n[self.I_minus].sum()
        self.n_samples = self.n_samples_plus + self.n_samples_minus
        self.n_features = mat.shape[1]
