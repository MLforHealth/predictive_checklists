import tempfile
import pandas as pd
try:
    import cplex
    from cplex import Cplex
    from cplex.exceptions import CplexError
except:
    print("CPLEX package not installed. Please download IBM ILOG CPLEX Optimization Studio here " + 
    'and then install the provided python package: https://www.ibm.com/products/ilog-cplex-optimization-studio')
    raise

import numpy as np
from IPChecklists.utils import set_seed, StatsCallback, binary_metrics, ProcessedDataset
from IPChecklists.constraints_cplex import *
print_vnames = lambda vfmt, vcnt: list(map(lambda v: vfmt % v, range(vcnt)))
from IPChecklists.Covers import CoverResult
from IPChecklists.Checklist import Checklist

print("Using CPLEX version " + cplex.__version__)

logging.basicConfig(level='INFO')
log = logging.getLogger()


def _add_variable(model, name, obj, lb, ub, vtype):

    name = [name] if type(name) is not list else name
    obj = [float(obj)] if type(obj) is not list else obj
    ub = [float(ub)] if type(ub) is not list else ub
    lb = [float(lb)] if type(lb) is not list else lb
    vtype = [vtype] if type(vtype) is not list else vtype

    assert isinstance(name[0], str)
    assert isinstance(obj[0], float)
    assert isinstance(ub[0], float)
    assert isinstance(lb[0], float)
    assert isinstance(vtype[0], str)

    model.variables.add(names = name, obj = obj, lb = lb, ub = ub, types = vtype)

def _add_constraint(model, lhs_ind, rhs_ind, sense = 'E', name = None):
    # sense = "G" -> lhs >= rhs

    if not isinstance(lhs_ind, list):
        lhs_ind = [lhs_ind]

    if not isinstance(rhs_ind, list):
        rhs_ind = [rhs_ind]

    assert len(lhs_ind) > 0 and len(rhs_ind) > 0

    con_args = {
        'senses': [sense],
        'lin_expr': [cplex.SparsePair(ind = lhs_ind + rhs_ind, val = [1.0] * len(lhs_ind) + [-1.0] * len(rhs_ind))],
        'rhs': [0.0],
        }

    if name is not None:
        con_args['names'] = [name]

    model.linear_constraints.add(**con_args)


class ChecklistMIP(object):
    def __init__(self, dataset, model_name = 'checklist', cost_func = '01',  one_feature_per_group = True,
                    lplus_weight = None, lminus_weight = None, compress = True):
        '''
            dataset: object of type BinaryDataset  
            cost_func: one of ['01', 'FNR', 'FPR'] 
            model_name: name of the checklist; only use for display purposes
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

    def build_problem(self, N_constraint = None, M_constraint = None, use_indicator = False):  
        '''
            Builds the MIP by adding constraints and the objective.

            Args:
                N_constraint, M_constraint: Passing in these constraints here instead of using .add_constraint() will improve solution efficiency.
                        Should be type MaxNumFeatureConstraint and MConstraint respectively.
                use_indicator: whether to use indicator constraints instead of big-M constraints. Probably does not make a big difference.
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

        model = Cplex()
        model.objective.set_sense(model.objective.sense.minimize)
        constraints = model.linear_constraints
        
        names = {
            'lamb': print_vnames('lamb%d', n_features), # 1 if checklist includes item j
            'z': print_vnames('z%d', n_samples), # 1 if mistake on sample i is predicted correctly
            'M': 'M',
            'N': 'N',
            'lplus': 'lplus', # errors on positive samples
            'lminus': 'lminus', # errors on negative samples
            'l': 'l'
            }
            
        # decision variables
        _add_variable(model, names['lamb'], [0.0] * n_features, [0.0] * n_features, [1.0] * n_features, ['B'] * n_features)
        _add_variable(model, names['M'], [self.regularizer2], [M_min], [M_max], ['I'])
        _add_variable(model, names['z'], [0.0] * n_samples, [0.0] * n_samples, [1.0] * n_samples, ['B'] * n_samples)

        # number of rules
        _add_variable(model, names['N'], [self.regularizer1], [N_min], [N_max], ['I'])
        _add_constraint(model, names['N'], names['lamb'], sense = 'E')

        # M < N
        _add_constraint(model, names['M'], names['N'], sense = 'L')

        # number of mistakes on positive/negative samples
        _add_variable(model, names['lplus'], [self.lplus_weight], [0.0], [float(n_samples_plus)], ['I']) # false negatives
        _add_variable(model, names['lminus'], [self.lminus_weight], [0.0], [float(n_samples_minus)], ['I']) # false positives
        _add_variable(model, names['l'], [0.0], [0.0], [float(min(n_samples_plus, n_samples_minus))], ['I'])

        constraints.add(lin_expr = [cplex.SparsePair(ind = [names['lplus']] + ['z%d' % i for i in I_plus], val = [1.0] + (-n[I_plus]).tolist())], senses = ['E'], rhs = [0.0])
        constraints.add(lin_expr = [cplex.SparsePair(ind = [names['lminus']] + ['z%d' % i for i in I_minus], val = [1.0] + (-n[I_minus]).tolist())], senses = ['E'], rhs = [0.0])
        _add_constraint(model, names['l'], [names['lplus'], names['lminus']], sense = 'E')
        
        if use_indicator:
            # positive samples: z[i] = 0 -> M - num_checked <= 0   
            for i in I_plus:
                model.indicator_constraints.add(
                    lin_expr = cplex.SparsePair(ind = [names['M']] + names['lamb'], val = [1.0] + (-mat[i, :]).tolist() ),
                    sense = 'L',
                    rhs = 0.0,
                    indvar = names['z'][i],
                    complemented = True                    
                )
            # negative examples: z[i] = 0 -> num_checked - M +1 <= 0
            for i in I_minus:
                model.indicator_constraints.add(
                    lin_expr = cplex.SparsePair(ind = [names['M']] + names['lamb'],  val = [-1.0] + mat[i, :].tolist()),
                    sense = 'L',
                    rhs = -1.0,
                    indvar = names['z'][i],
                    complemented = True                    
                )
                
        else:
            # compute B
            B = np.zeros(n_samples)
            B[I_plus] = M_max + 1 #todo: this can be tighter I think... work from max(M - \sum(lambda[j]x[i,j]) over M & lambda.
            B[I_minus] = np.minimum(mat[I_minus, :].sum(axis = 1), N_max) - M_min + 2
            B = B.tolist()

            # z_i = 1 if mistake on negative example
            for i in I_minus:
                # B[i]*z[i] + M - num_checked >= 1
                constraints.add(lin_expr = [cplex.SparsePair(ind = [names['z'][i], names['M']] + names['lamb'], val = [B[i], 1.0] + (-mat[i, :]).tolist())],
                                senses = ['G'],
                                rhs = [1.0])

            # z_i = 1 if mistake on positive example
            for i in I_plus:
                # B[i]*z[i] - M + num_checked >= 0       
                constraints.add(lin_expr = [cplex.SparsePair(ind = [names['z'][i], names['M']] + names['lamb'], val = [B[i], -1.0] + mat[i, :].tolist())],
                                senses = ['G'],
                                rhs = [0.0])


        # constraints
        for con in self.constraints:
            con_formulation = con.add(names, self.dataset, self.binarized_df, )
            if 'SOS' in con_formulation:
                model.SOS.add(type = "1", **con_formulation)
            else:           
                model.linear_constraints.add(**con_formulation)

        if self.one_feature_per_group: # add one-feature-per-group constraints
            for ft_name in self.dataset.feature_mapping:
                con = FeaturesPerGroupConstraint(ft_name, '<=', 1)
                con_formulation = con.add(names, self.dataset, self.binarized_df, )
                model.SOS.add(type = "1", **con_formulation)
                
        # add complement-pair constraints
        for ft_name in self.dataset.feature_mapping:
            for n1, n2 in self.dataset.feature_mapping[ft_name].complement_pairs:
                con = MostOneofTwoConstraint(n1, n2)
                con_formulation = con.add(names, self.dataset, self.binarized_df)
                if con_formulation:
                    model.SOS.add(type = "1", **con_formulation)
                
        # add the two constraints passed to this function to named constraints
        if M_constraint is not None:
            self.add_constraint(M_constraint)
            
        if N_constraint is not None:
            self.add_constraint(N_constraint)

        self.model = model
        self.names = names
        self.N_min, self.N_max, self.M_min, self.M_max = N_min, N_max, M_min, M_max
        self.built = True

    def add_constraint(self, constraint):
        '''Adds a constraint to the MIP. Constraints can be found in constraints_cplex'''
        assert isinstance(constraint, Constraint)
        assert not self.built, "Constraints need to be added before building the MIP."
        self.constraints.append(constraint)

    def solve(self, max_seconds = None, threads = 1, return_incumbents = False,
              display_progress = False, emphasis = 1, variableselect = 3,
              memory_limit = 1024*20, seed = 42):

        '''
            Solves the MIP. 

            Args: 
                max_seconds: maximum number of seconds to solve for
                threads: number of threads to use
                return_incumbents: whether to store and return incumbent solutions
                display_progress: print progress log to the screen
                emphasis, variableselect: see https://www.ibm.com/docs/en/SSSA5P_12.8.0/ilog.odms.studio.help/pdf/usrcplex.pdf
                memory_limit: memory limit in MB
                seed: random seed
        '''

        assert self.built, "Must build MIP first!"

        if self.solved:
            log.warn('problem is already solved!')
            return

        # parameters
        set_seed(seed)
        self.model.parameters.randomseed.set(seed)
        self.model.parameters.mip.strategy.file = 3 #nodefile compressed
        self.model.parameters.workdir = tempfile.gettempdir()
        self.model.parameters.emphasis.mip = emphasis
        self.model.parameters.mip.strategy.variableselect = variableselect
        self.model.parameters.mip.limits.treememory = memory_limit # https://www.ibm.com/support/pages/apar/RS03251

        if max_seconds is not None:
            self.model.parameters.timelimit.set(max_seconds)

        self.model.parameters.threads.set(threads)
        if display_progress == False:
            self.model.parameters.mip.display.set(0)
            self.model.set_results_stream(None)
            self.model.set_log_stream(None)

        sol_idx = self.model.variables.get_indices(self.names['lamb'] + [self.names['M']])
        self._add_stats_callback(sol_idx, store_solutions = return_incumbents)
        self.model.solve()

        solution_info = self.get_solution_info()
        if solution_info['iterations'] > 0:
            progress_info, progress_incumbents = self._stats_callback.get_stats()
        else:
            progress_info, progress_incumbents = [], []

        if solution_info['has_solution']:
            self.solved = True
            print(f"Found solution with objective {solution_info['objval']} and optimality gap {solution_info['gap']:.2%}.")
            self.lamb = np.array(self.model.solution.get_values(self.names['lamb']))
            self.M = self.model.solution.get_values(self.names['M'])
            self.N = self.model.solution.get_values(self.names['N'])
        else:
            log.warn('No feasible solution found')

        return solution_info, progress_info, progress_incumbents

    def _add_stats_callback(self, sol_idx, store_solutions = False):
        if not hasattr(self, '_stats_callback'):
            min_idx = min(sol_idx)
            max_idx = max(sol_idx)
            assert np.array_equal(np.array(sol_idx), np.arange(min_idx, max_idx + 1))

            cb = self.model.register_callback(StatsCallback)
            cb.initialize(store_solutions, solution_start_idx = min_idx, solution_end_idx = max_idx)
            self._stats_callback = cb

    def get_solution_info(self):
        """returns information associated with the current best solution for the mip"""
        INITIAL_SOLUTION_INFO = {
            'status': 'no solution exists',
            'status_code': float('nan'),
            'has_solution': False,
            'has_mipstats': False,
            'iterations': 0,
            'nodes_processed': 0,
            'nodes_remaining': 0,
            'values': float('nan'),
            'objval': float('nan'),
            'upperbound': float('nan'),
            'lowerbound': float('nan'),
            'gap': float('nan'),
            }

        info = dict(INITIAL_SOLUTION_INFO)
        try:
            sol = self.model.solution
            progress_info = {'status': sol.get_status_string(),
                             'status_code': sol.get_status(),
                             'iterations': sol.progress.get_num_iterations(),
                             'nodes_processed': sol.progress.get_num_nodes_processed(),
                             'nodes_remaining': sol.progress.get_num_nodes_remaining()}
            info.update(progress_info)
            info['has_mipstats'] = True
        except CplexError:
            pass

        try:
            sol = self.model.solution
            solution_info = {'values': np.array(sol.get_values()),
                             'objval': sol.get_objective_value(),
                             'upperbound': sol.MIP.get_cutoff(),
                             'lowerbound': sol.MIP.get_best_objective(),
                             'gap': sol.MIP.get_mip_relative_gap()}
            info.update(solution_info)
            info['has_solution'] = True
        except CplexError:
            pass

        return info

    def add_initial_solution_from_mip(self, other_mip):
        '''Adds an initial solution to the MIP in the form of a solved MIP.'''
        assert(isinstance(other_mip, ChecklistMIP))
        var_names = other_mip.model.variables.get_names()
        var_values = other_mip.model.solution.get_values(var_names)
        self.model.MIP_starts.add([var_names, var_values], self.model.MIP_starts.effort_level.check_feasibility)    
        
    def add_initial_solution_from_checklist(self, checklist):
        '''Adds an initial solution to the MIP in the form of a checklist.'''
        assert(isinstance(checklist, Checklist))
        for i in checklist.column_names:
            if i not in self.remaining_features:
                log.warn(f'Trying to add initial solution with different feature: {i}. Skipping.')
                return None
        self.add_initial_solution_from_weights(checklist.lamb, checklist.M) 
        
    def add_initial_solution_from_weights(self, lamb, M):
        '''Adds an initial solution to the MIP by providing the lambda vector and M directly.'''
        assert(len(lamb) == len(self.names['lamb']))
        assert(M <= self.M_max and M >= self.M_min), (M, self.M_max, self.M_min)
        assert(i in [0, 1, 0., 1.] for i in lamb)
        N_init = sum(lamb)
        assert(N_init <= self.N_max and N_init >= self.N_min), (N_init, self.N_max, self.N_min)
        
        var_names = self.names['lamb'] + [self.names['M'], self.names['N']]
        var_values = [float(i) for i in lamb] + [float(M), float(N_init)]
        self.model.MIP_starts.add([var_names, var_values], self.model.MIP_starts.effort_level.solve_MIP)   

    def pretty_print_soln(self):
        string = ''
        string += ('-' * 30) + '\n'
        string += ('Model name: ' + self.model_name ) + '\n'
        string += ('Regularizer1: ' + str(self.regularizer1) ) + '\n'
        string += ('Regularizer2: ' + str(self.regularizer2) ) + '\n'
        string += ('Constraints: ' + "None" + '\n' if len(self.constraints) == 0 else '\n'+ '\n'.join(str(i) for i in self.constraints) + '\n')
        string += ('-' * 30) + '\n'
        string += ('Training loss: %s' % (self.model.solution.get_objective_value())) + '\n'
        string += ('Num errors: %s' % self.model.solution.get_values(self.names['l'])) + '\n'
        string += (f"Training optimality gap: {self.model.solution.MIP.get_mip_relative_gap():.2%}") + '\n'
        string += ('-' * 30) + '\n'
        for v, col in zip(self.lamb, self.binarized_df.columns):
            if v == 1:
                string += (col) + '\n'
                
        string += '\n'
        string += ('M = %s, N = %s'%(self.M,  self.N)) + '\n'
        string += ('-' * 30)+ '\n'
        return string   
                
    def get_progress_logs(self):
        return self._stats_callback.get_stats()[0]

    def plot_progress_logs(self):
        plogs = self.get_progress_logs()
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(1, 2)
        ax = axs[0]
        ax.plot(plogs['runtime'], plogs['lowerbound'], '-', label = 'lb')
        ax.plot(plogs['runtime'], plogs['upperbound'], '-', label = 'ub')
        ax.legend()
        ax.set_xlabel('t (s)')
        ax.set_ylabel('Objective')
                
        ax = axs[1]
        ax.plot(plogs['nodes_processed'], plogs['lowerbound'], '-', label = 'lb')
        ax.plot(plogs['nodes_processed'], plogs['upperbound'], '-', label = 'ub')
        ax.legend()
        ax.set_xlabel('# Nodes')
        ax.set_ylabel('Objective')
        
        plt.tight_layout()
        plt.show()
        
    def to_checklist(self):
        return Checklist(from_mip = True, mip = self)