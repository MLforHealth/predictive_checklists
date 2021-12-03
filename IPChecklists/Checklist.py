import numpy as np
import pandas as pd
import logging
logging.basicConfig(level='INFO')
log = logging.getLogger()
from IPChecklists.utils import binary_metrics
import re

class Checklist(object):
    """A Checklist object. Instantiated from a trained MIP (ChecklistMIP.to_checklist), Submodular cover (CoverResult.to_checklist), or unit weighting."""
    def __init__(self, from_mip = False, from_weights = False, mip = None, lamb = None, M = 0, column_names = []):
        self.from_mip = from_mip
        if from_mip:
            if mip.solved:          
                self.is_valid = True
                self.lamb = mip.lamb.round()
                self.M = mip.M
                self.N = self.lamb.sum()
                self.column_names = mip.remaining_features
                
                # only for mips
                self.progress_log = mip.get_progress_logs()
                self.checklist_str = mip.pretty_print_soln()
                self.solution_info = mip.get_solution_info()                
            else:
                self.is_valid = False
                log.warn("MIP with no solution passed into Checklist constructor!")
        elif from_weights:
            self.lamb = np.array(lamb)
            self.M = M        
            self.is_valid = True
            self.N = int(self.lamb.sum())
            self.column_names = column_names
        else:
            raise ValueError('No valid input data passed!')    

    def get_num_checked(self, dataset):
        """Returns an array of integers for how many items are checked off for each sample."""
        if not self.is_valid:
            raise ValueError("Prediction with invalid checklist!")      
        df_bin = dataset.binarized_df
        if self.from_mip:
            mat = df_bin[self.column_names].values
        else:
            mat = df_bin.values
        return mat @ self.lamb

    def predict(self, dataset):
        return self.get_num_checked(dataset) >= self.M    
    
    def get_metrics(self, ds, return_arrays = False):
        ''' Returns performance metrics of this checklist for a dataset.
        Args:
            ds: BinaryDataset
            return_arrays: Whether to include target and prediction arrays in the output        
        '''
        target = np.copy(ds.target)
        target[target == -1] = 0
        
        preds = self.predict(ds)
        res = binary_metrics(target, preds, return_arrays)
        return res    

    def get_metrics_from_df(self, df_bin, target):
        target = np.copy(target)
        target[target == -1] = 0
        if self.from_mip:
            mat = df_bin[self.column_names].values
        else:
            mat = df_bin.values

        preds = (mat @ self.lamb) >= self.M    
        res = binary_metrics(target, preds)
        return res   

    def get_fairness_metrics(self, ds, attributes):
        rows = {}
        for attr in attributes:        
            for grp in ds.protected_groups[attr].unique():
                df_bin_i, target_i = ds.subset_by_protected_group(attr, grp)
                metrics_i = self.get_metrics_from_df(df_bin_i, target_i)
                rows[grp] = metrics_i
        return pd.DataFrame(rows)
    
    def __repr__(self):
        return self.pretty_print_soln()

    def pretty_print_soln(self):
        string = ''
        for i in range(len(self.lamb)):
            if self.lamb[i] == 1:
                string += (self.column_names[i]) + '\n'
        string += '\n'
        string += ('M = %s, N = %s'%(self.M,  self.N)) + '\n'
        return string

    def get_selected_cols(self):
        return [self.column_names[c] for c, i in enumerate(self.lamb) if i == 1]
    
    def __eq__(self, other):
        return self.lamb == other.lamb and self.M == other.lamb and self.column_names == other.column_names

    def to_latex(self, target_name):
        st = r'\begin{tabular}[c]{@{}>{\sffamily}l@{}r}' + '\n'
        st += f'Predict {target_name} if {int(self.M)}+ Items are Checked\\\\\n\\hline\n'
        for col in self.get_selected_cols():
            col = col.replace('==', '=').replace('_', '\_')
            if col.count('=') <= 1 and col.count('~') <= 1:
                col = col.replace('~', '=')
                p = re.compile(r'(?P<compr>[<>!]?=)')
                col = p.sub(' $\g<compr>$ ', col).replace('<=', '\\leq').replace('>=', '\\geq')
            st+= col + r'& $\square$\\' + '\n'
        st += '\end{tabular}\n'
        return st
