## Formulation of the problem using PythonMIP - currently not updated
from mip import Model, BINARY, INTEGER, minimize, xsum, compute_features, features, INF, CBC, GRB
import warnings
warnings.warn("Loaded the Python MIP interface. We provide NO GUARANTEES on the performance of the Python MIP solvers. " + 
        "Where possible, please use CPLEX instead.")
from tqdm import tqdm
import numpy as np
import random
from .utils import set_seed
import pandas as pd
import logging
import os
from IPChecklists.utils import ProcessedDataset
from IPChecklists.constraints_pythonmip import *
from IPChecklists.Checklist import Checklist

logging.basicConfig(level='INFO')
log = logging.getLogger()
    
class ChecklistMIP():
    def __init__(self, dataset, model_name = 'checklist', cost_func = '01', solver = 'CBC',
                    one_feature_per_group = True, lplus_weight = None, lminus_weight = None, compress = True):
        '''
            dataset: object of type BinaryDataset  
            cost_func: one of ['01', 'balanced_01', 'FNR', 'FPR'] 
            model_name: name of the checklist; only use for display purposes
            solver: Python-MIP solver; 'CBC' or 'GRB'
            one_feature_per_group: whether to enforce the constraint to select at most one variable from each feature group
            lplus_weight, lminus_weight: custom weights for positive and negative errors, overrides cost_func weights
            compress: whether to compress samples that have the same features. Must be False for fairness constraints.
        '''
        self.dataset = dataset
        target = dataset.target.copy()
        target[target == 0] = -1                    
        self.binarized_df = dataset.binarized_df.copy() 
        self.remaining_features = dataset.remaining_features
        
        self.target = target.values
        self.model_name = model_name
        self.cost_func = cost_func
        self.constraints = [] # named constraints
        self.solved = False     
        self.solver = solver
        self.built = False 

        self.prod = self._init(compress = compress) # processed dataset

        if cost_func == '01':
            self.lplus_weight = 1.0
            self.lminus_weight = 1.0

        elif cost_func == 'FPR': # minimize false positives = minimize errors on negative samples
            self.lplus_weight = 1.0 
            self.lminus_weight = self.prod.n_samples_minus + 0.5

        elif cost_func == 'FNR': # 
            self.lplus_weight = self.prod.n_samples_plus + 0.5
            self.lminus_weight = 1.0 

        elif cost_func == 'balanced_01':
            self.lplus_weight = 1.0 # 1.0/self.prod.n_samples_plus
            self.lminus_weight = self.prod.n_samples_plus/self.prod.n_samples_minus # 1.0/self.prod.n_samples_minus
        else:
            raise NotImplementedError(cost_func)

        self.lplus_weight = lplus_weight if lplus_weight is not None else self.lplus_weight      
        self.lminus_weight = lminus_weight if lminus_weight is not None else self.lminus_weight
        
        self.regularizer1 = min(self.lplus_weight, self.lminus_weight) * 0.95 / float(self.binarized_df.shape[0]) / float(len(self.dataset.feature_mapping))
        self.regularizer2 = min(self.lplus_weight, self.lminus_weight) * 0.95 * self.regularizer1 / float(len(self.dataset.feature_mapping))
        self.one_feature_per_group = one_feature_per_group    
    
    def _init(self, compress = True):
        # compress data
        df = pd.DataFrame(self.binarized_df)
        df['target'] = self.target
        if compress:
            log.warn('Do not use compression when using fairness constraints!')
            log.info("Before compression: %s rows" % df.shape[0])        
            df = df.groupby(df.columns.tolist()).size().reset_index(name='weight_i')
            log.info("After compression: %s rows" % df.shape[0])
        else:
            df = df.reset_index(drop = True)
            df['weight_i'] = 1

        n = df['weight_i'].values
        target = df['target'].values
        I_plus = df.query('target == 1').index.tolist()
        I_minus = df.query('target == -1').index.tolist()
        df = df.drop(columns = ['target', 'weight_i'])
        mat = df.values

        return ProcessedDataset(mat, target, I_plus, I_minus, n)
    
    def build_problem(self, N_constraint = None, M_constraint = None, *args, **kwargs):
        '''
            Builds the MIP by adding constraints and the objective.

            Args:
                N_constraint, M_constraint: Passing in these constraints here instead of using .add_constraint() will improve solution efficiency.
                        Should be type MaxNumFeatureConstraint and MConstraint respectively.
        '''
        (n_features, n_samples, n_samples_plus, n_samples_minus, n, I_plus, I_minus, mat) = (self.prod.n_features, 
            self.prod.n_samples, self.prod.n_samples_plus, self.prod.n_samples_minus, self.prod.n, self.prod.I_plus, self.prod.I_minus, self.prod.mat)
        
        if N_constraint is not None:
            assert isinstance(N_constraint, MaxNumFeatureConstraint)
            N_min, N_max = N_constraint.get_min_max(n_features)
        else:
            N_min, N_max = 1.0, float(n_features)
            
        if M_constraint is not None:
            assert isinstance(M_constraint, MConstraint)
            M_min, M_max = M_constraint.get_min_max(n_features)
        else:
            M_min, M_max = 0.0, float(n_features)        
        
        M_max = min(M_max, N_max)

        model = Model(self.model_name, solver_name = self.solver)    
        model.store_search_progress_log = True
        model.preprocess = 1
        model.start = []
                
        # decision variables
        lamb = [model.add_var(var_type=BINARY) for i in range(n_features)]
        z = [model.add_var(var_type=BINARY) for i in range(n_samples)]
        M = model.add_var(var_type=INTEGER, lb = 1, ub = n_features)
        
        # number of rules
        N = xsum(lamb)
        model += N <= N_max
        
        # constrain M
        model += M - N <= 0
        model += M <= M_max
                
        # number of mistakes on positive/negative samples
        lplus = xsum(n[i]*z[i] for i in I_plus)
        lminus = xsum(n[i]*z[i] for i in I_minus)
        l = lplus + lminus
        
        B = np.zeros(n_samples)
        B[I_plus] = M_max + 1
        B[I_minus] = np.minimum(mat[I_minus, :].sum(axis = 1), N_max) - M_min + 2
        B = B.tolist()

        # z_i = 1 if mistake on positive example
        for i in I_plus:
            model += ( 
                B[i]*z[i] >= M - xsum(lamb[j]*mat[i, j] for j in range(n_features))                
            )
        
        # z_i = 1 if mistake on negative example
        for i in I_minus:
            model += ( 
                B[i]*z[i] >= xsum(lamb[j]*mat[i, j] for j in range(n_features)) - M + 1
            )
                
        # cost function
        model.objective = minimize(self.lminus_weight*lminus + self.lplus_weight*lplus 
                + self.regularizer1 * N + self.regularizer2 * M)  
         
        # collect variables
        self.m_vars = {
            'model': model,
            'mat': mat,
            'n': n,
            'I_plus': I_plus,
            'I_minus': I_minus,
            'B': B,
            'lamb': lamb,
            'z': z,
            'M': M,
            'N': N,
            'lplus': lplus,
            'lminus': lminus,
            'l': l
        }
        
        # constraints
        for con in self.constraints:
            con.add(self.m_vars, self.dataset, self.binarized_df)         
            
        if self.one_feature_per_group: # add one-feature-per-group constraints
            for ft_name in self.dataset.feature_mapping:
                con = FeaturesPerGroupConstraint(ft_name, '<=', 1)
                con.add(self.m_vars, self.dataset, self.binarized_df)
                
        # add complement-pair constraints
        for ft_name in self.dataset.feature_mapping:
            for n1, n2 in self.dataset.feature_mapping[ft_name].complement_pairs:
                con = MostOneofTwoConstraint(n1, n2)
                con.add(self.m_vars, self.dataset, self.binarized_df)

        # add the two constraints passed to this function to named constraints
        if M_constraint is not None:
            self.add_constraint(M_constraint)
            
        if N_constraint is not None:
            self.add_constraint(N_constraint)

        self.N_min, self.N_max, self.M_min, self.M_max = N_min, N_max, M_min, M_max
        self.built = True
    
    def add_constraint(self, constraint):
        '''Adds a constraint to the MIP. Constraints can be found in constraints_pythonmip'''
        assert(isinstance(constraint, Constraint))
        assert not self.built, "Constraints need to be added before building the MIP."
        self.constraints.append(constraint)     

    def solve(self, max_seconds = INF, threads = 1, 
        display_progress = False, emphasis = 1, seed = 42):  
        '''
            Solves the MIP. 

            Args: 
                max_seconds: maximum number of seconds to solve for
                threads: number of threads to use
                emphasis: see https://python-mip.readthedocs.io/en/latest/classes.html#mip.SearchEmphasis
                seed: random seed
        '''      
        assert self.built, "Must build MIP first!"        
        
        if self.solved:
            log.warn('problem is already solved!')
            return         
        
        self.m_vars['model'].threads = threads
        self.m_vars['model'].emphasis = emphasis
        self.m_vars['model'].seed = seed
        self.m_vars['model'].verbose = int(display_progress)
        
        self.status = self.m_vars['model'].optimize(max_seconds = max_seconds)
        if self.m_vars['model'].num_solutions:
            self.solved = True
            print(f"Found solution with objective {self.m_vars['model'].objective_value} and optimality gap {self.get_optimality_gap():.2%}.")
            self.lamb = np.array([self.m_vars['lamb'][i].x for i in range(len(self.m_vars['lamb']))])   
            self.M = self.m_vars['M'].x
            self.N = self.m_vars['N'].x
        else:
            log.warn('No feasible solution found')

        return self.get_solution_info(), self.get_progress_logs(), []
    
    def get_optimality_gap(self):
        temp = self.get_progress_logs().iloc[-1]
        if temp['ub'] == 0.0:
            return 1
        else:            
            return (temp['ub'] - temp['lb'])/temp['ub'] 

    def add_initial_solution_from_checklist(self, checklist):
        '''Adds an initial solution to the MIP in the form of a checklist.'''
        assert(isinstance(checklist, Checklist))
        for i in checklist.column_names:
            if i not in self.remaining_features:
                log.warn(f'Trying to add initial solution with different feature: {i}. Skipping.')
                return None
        self.add_initial_solution_from_weights(checklist.lamb, checklist.M) 

    def add_initial_solution_from_weights(self, lamb, M):
        assert(len(lamb) == len(self.m_vars['lamb']))
        assert(M <= self.M_max and M >= self.M_min), (M, self.M_max, self.M_min)
        assert(i in [0, 1, 0., 1.] for i in lamb)
        N_init = sum(lamb)
        assert(N_init <= self.N_max and N_init >= self.N_min), (N_init, self.N_max, self.N_min)
        
        self.m_vars['model'].start = [(self.m_vars['M'], M)] + [(a,b) for a,b in zip(self.m_vars['lamb'], lamb)] 
        
    def pretty_print_soln(self):
        string = ''
        string += ('-' * 30) + '\n'
        string += ('Model name: ' + self.model_name ) + '\n'
        string += ('Regularizer1: ' + str(self.regularizer1) ) + '\n'
        string += ('Regularizer2: ' + str(self.regularizer2) ) + '\n'
        string += ('Constraints: ' + "None" + '\n' if len(self.constraints) == 0 else '\n'+ '\n'.join(str(i) for i in self.constraints) + '\n')
        string += ('-' * 30) + '\n'
        string += ('Training loss: %s' % self.m_vars['model'].objective_value) + '\n'
        string += (f"Training optimality gap: {self.get_optimality_gap():.2%}") + '\n'
        string += ('-' * 30) + '\n'
        for i in range(len(self.m_vars['lamb'])):
            if self.m_vars['lamb'][i].x == 1:
                string += (self.binarized_df.columns[i]) + '\n'
                
        string += '\n'
        string += ('M = %s, N = %s'%(self.m_vars['M'].x, self.m_vars['N'].x)) + '\n'
        string += ('-' * 30)+ '\n'
        return string   
    
    def print_properties(self):
        a,b = compute_features(self.m_vars['model']), features()
        for x,y in zip (a,b):
            print('%s: %s'% (y, x))
            
    def get_progress_logs(self):
        temp = pd.DataFrame(self.m_vars['model'].search_progress_log.log, columns = ['t', 'data'])
        temp['lb'] = temp['data'].apply(lambda x: x[0])        
        temp['ub'] = temp['data'].apply(lambda x: x[1])
        return temp.drop(columns = ['data'])

    def get_solution_info(self):
        return {
            'gap': float(self.m_vars['model'].gap),
            'lowerbound': float(self.m_vars['model'].objective_bound),
            'upperbound': float(self.m_vars['model'].objective_value)
        }

    def to_checklist(self):
        return Checklist(from_mip = True, mip = self)