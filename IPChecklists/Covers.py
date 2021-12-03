import numpy as np
import pandas as pd
import copy
from IPChecklists.Checklist import Checklist
import math
from tqdm import tqdm

def cardinality_with_matroid(S, K, f, matroid = None, *args, **kwargs):
    '''
        S (list): list of feature indices
        K (int): cardinality constraint
        f (function): submodular function to maximize, should be parallelized across samples
        matroid (function): function that takes in a set of indices 
            and returns whether matroid is satisfied
        other args: passed to f (ex: M)
    '''
    all_sgs, all_func_vals = [], []
    sg = []
    while len(sg) < K and len(S) > 0:
        func_vals = f(sg, S, *args, **kwargs)
        if matroid is not None:
            satisfies_p = np.array(list(map(lambda x: matroid(sg + [x]), S)))
        else:
            satisfies_p = np.array([True] * len(S))

        x_star_ind = np.arange(len(S))[np.argmax(func_vals)]
        if satisfies_p[x_star_ind]:
            sg.append(S[x_star_ind]) 
            if len(all_func_vals) > 0 and func_vals[x_star_ind] == all_func_vals[-1]: # already maximized function
                break

            all_sgs.append(copy.deepcopy(sg))
            all_func_vals.append(func_vals[x_star_ind]) 

        if isinstance(S, np.ndarray):        
            S = np.delete(S, x_star_ind)
        else:
            del S[x_star_ind]

    return (all_sgs, all_func_vals)      


def knapsack_with_p_system(S, K, f, c, budget, eps, p_system = None, f_parallel = None, *args, **kwargs):
    '''
       Algorithm 10 from "Fast algorithms for maximizing submodular functions" (Badanidiyuru and Vondrak, 2013)

       Args:
            S (list): list of feature indices
            K (int): cardinality constraint
            f (function): submodular function to maximize    
            c (list): cost of each item for knapsack constraint
            budget (float): budget for knapsack constraint
            eps (float): step size for algorithm
            p_system (function): function that takes in a set of indices 
                and returns whether p system is satisfied
            f_parallel (function): f but parallelized over some batch size
            other args: passed to f and f_parallel (ex: M)
    '''
    assert((S == np.arange(len(S))).all())
    all_sgs, all_func_vals = [], []
    if f_parallel is not None:
        m = max(f_parallel([], S, *args, **kwargs))
    else:
        m = max(list(map(lambda x: f([x], *args, **kwargs), S)))
    p = 2 if p_system is not None else 1 # cardinality + partition matroid
    l = 1 # one knapsack 
    n = len(S)
    c = c/float(budget)  
                
    rho_seq = [m/(p+l)]
    while rho_seq[-1] < 2*m*n/(p+l):
        rho_seq.append((1+eps)*rho_seq[-1])    
                
    rho_results = {}    
    for rho in rho_seq:
        if f_parallel is not None:
            tau_list = f_parallel([], S, *args, **kwargs)
        else:
            tau_list = list(map(lambda x: f([x], *args, **kwargs), S))
        tau_list = [tau_list[j] for j in range(len(S)) if c[j] == 0 or tau_list[j]/c[j] >= rho]
        if len(tau_list) == 0:
            break
                    
        tau = max(tau_list)        
        sg = []
        tp, tpp = None, None
        
        restart_next_rho = False        
        
        while tau >= eps * m/n and compute_cost(c, sg) <= 1 and not restart_next_rho:
            if f_parallel is not None:
                fs = f_parallel(sg, S, *args, **kwargs) - f(sg, *args, **kwargs)
                
            for j_ind, j in enumerate(S):
                if f_parallel is None:
                    fs_j = f(sg + [j], *args, **kwargs) - f(sg, *args, **kwargs)
                else:
                    fs_j = fs[j_ind]
                
                if ((p_system is None or p_system(sg + [j])) 
                        and fs_j  >= tau 
                        and (c[j_ind] == 0 or fs_j / c[j_ind] >= rho)
                        and len(sg) + 1 <= K) :
                    sg.append(j)
                    if compute_cost(c, sg) > 1:
                        tp = sg[:-1]
                        tpp = [sg[-1]]
                        restart_next_rho = True                        
                        break
                    else:
                        # recompute fs since sg is now different
                        if f_parallel is not None:
                            fs = f_parallel(sg, S, *args, **kwargs) - f(sg, *args, **kwargs)                        
                        
            tau *= 1/(1+eps)            
        
        if not restart_next_rho:
            tp = sg
        
        if tp is not None and len(tp) > 0: 
            rho_results[rho] = [tp]
            if tpp is not None and len(tpp) > 0:
                rho_results[rho].append(tpp)
                
    return rho_results        
    
def compute_cost(c, sg):
    return np.take(c, sg).sum()

def query_function(S, M, X):
    # s is a list of indices of features
    weights = np.zeros((X.shape[1],))
    weights[S] = 1
    return np.sum(np.minimum(np.dot(X, weights), M))

def query_function_parallel(sg, S, M, X):
    # sg (list): list of features indices currently in the set
    # S (list): list of feature indices that can be added
    weights = np.zeros((X.shape[1], len(S)))
    weights[sg, :] = 1
    weights[S, np.arange(weights.shape[1])] = 1    
    return np.sum(np.minimum(np.dot(X, weights), [M] * len(S)), axis = 0)

def satisfy_constraints(s, splits):
    '''
        s (list): subset of features to check
        splits (list of sets): allow at most one feature from each set
    '''    
    s = set(s)
    for orig in splits:
        if len(orig.intersection(s)) > 1:
            return False
    return True   

def remove_duplicates(results):
    indices_sets, result_covers = [], []
    for cover_result in results:
        index_set = set(list(cover_result.indices))
        if index_set not in indices_sets and cover_result.M <= cover_result.N:
            indices_sets.append(index_set)
            result_covers.append(cover_result)
    return result_covers

class Cover():
    def __init__(self, ds, M, max_N = None, cover_label = 1, enforce_group_constraint = False):
        super().__init__()
        self.ds = ds        
        self.M = M
        self.max_N = len(ds.binarized_df.shape[1]) if max_N is None else max_N
        self.cover_label = cover_label
        assert self.cover_label in [-1, 1, -1.0, 1.0]
        self.enforce_group_constraint = enforce_group_constraint
    
    def get_results(self, return_intermediate = True):
        if return_intermediate:
            return self.results
        return self.results[-1]

    def get_X(self):
        if self.cover_label == 1:
            X = self.ds.binarized_df[self.ds.target == 1]      
        elif self.cover_label == -1:  
            X = -self.ds.binarized_df[self.ds.target == -1] + 1
        else:
            raise ValueError
        return X.values

    def get_matroid(self):
        if self.enforce_group_constraint:
            splits = []
            for feature_set in self.ds.feature_mapping.values():
                inds = set()
                for c, i in enumerate(self.ds.binarized_df.columns):
                    if i in feature_set.binarized_names:
                        inds.add(c)
                splits.append(inds)
            matroid = lambda s: satisfy_constraints(s, splits)
        else:
            matroid = None
        return matroid


class SimpleCover(Cover):
    '''Submodular cover with cardinality constraint with/without p-system constraints'''
    def __init__(self, ds, M, max_N = None, cover_label = 1, enforce_group_constraint = True):       
        '''
            ds: training dataset
            M: number of items to check off to give a positive prediction
            max_N: maximum number of items to select (cardinality constraint)
            cover_label: Whether to cover the positive or the negative class. Should be +1 or -1 (int).
            enforce_group_constraint: whether to enforce "one from each feature group" as a p-system
        ''' 
        super().__init__(ds, M, max_N, cover_label, enforce_group_constraint)
        
    def solve(self):
        '''Solves the cover. Returns a list of CoverResult objects.'''
        X = self.get_X()
        matroid = self.get_matroid()
            
        sgs, vals = cardinality_with_matroid(np.arange(X.shape[1]), self.max_N, query_function_parallel,
                                    matroid = matroid, M = self.M, X = X)
        max_val = np.max(vals)
        upper_limit_orig = int(max_val / (1-1/math.e))
        # number of positive/negatives that can be covered in the original MIP
        upper_limit = max(0, upper_limit_orig - (self.M - 1) * (self.ds.target == self.cover_label).sum())            
        self.results = remove_duplicates([CoverResult(sg, self.M, val, upper_limit, self.ds) for sg, val in zip(sgs, vals)])       
        return self.results   

class KnapsackCover(Cover):
    '''Submodular cover with knapsack constraint + cardinality constraint with/without p-system constraints'''
    def __init__(self, ds, M, budget, eps, max_N = None, cover_label = 1, enforce_group_constraint = True):
        '''
            ds: training dataset
            M: number of items to check off to give a positive prediction
            budget: Budget for the knapsack constraint (float)
            eps: step size parameter from Algorithm 10 in (Badanidiyuru and Vondrak, 2013). 
            max_N: maximum number of items to select (cardinality constraint)
            cover_label: Whether to cover the positive or the negative class. Should be +1 or -1 (int).
            enforce_group_constraint: whether to enforce "one from each feature group" as a p-system
        '''
        super().__init__(ds, M, max_N, cover_label, enforce_group_constraint)
        self.budget = budget
        self.eps = eps
        self.p = int(enforce_group_constraint)
        
    def solve(self):
        '''Solves the cover. Returns a list of CoverResult objects.'''
        X = self.get_X()                
        c = (self.ds.binarized_df.values[self.ds.target == -self.cover_label] == 1).sum(axis = 0).astype(float)
            
        ratio = self.p + 2*2 + 1 + self.eps # l = 2
        p_system = self.get_matroid()

        results = knapsack_with_p_system(np.arange(X.shape[1]), self.max_N, query_function,
                                        c = c, budget = self.budget, eps = self.eps, p_system = p_system,
                                                f_parallel = query_function_parallel, M = self.M, X = X)
        sgs, vals = [], []
        for rho in results:
            for lst in results[rho]:
                if len(lst) > 0:
                    sgs.append(lst)
                    vals.append(query_function(lst, self.M, X))
        max_val = np.max(vals)            
        upper_limit_orig = int(max_val * ratio)
        # number of positive/negatives that can be covered in the original MIP
        upper_limit = max(0, upper_limit_orig - (self.M - 1) * (self.ds.target == self.cover_label).sum()) 
        
        self.results = remove_duplicates([CoverResult(sg, self.M, val, upper_limit, self.ds) for sg, val in zip(sgs, vals)])
        return self.results

class CoverResult():
    def __init__(self, indices, M, func_val, upper_limit, ds):
        self.N = len(indices)
        self.indices = indices
        self.lamb = np.zeros((ds.binarized_df.shape[1],))
        self.lamb[indices] = 1
        self.func_val = func_val
        self.M = M
        self.upper_limit = upper_limit
        self.column_names = ds.binarized_df.columns
        
    def to_checklist(self):
        '''Creates a Checklist from the cover result.'''
        return Checklist(from_weights = True, lamb = self.lamb, M = self.M, column_names = self.column_names)

    def __eq__(self, other):
        return set(list(self.indices)) == set(list(other.indices))