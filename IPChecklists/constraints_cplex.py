import cplex
import logging
import numpy as np

logging.basicConfig(level='INFO')
log = logging.getLogger()

def get_sense(op):
    if op == '<=':
        sense = 'L'
    elif op == '==':
        sense = 'E'
    elif op == '>=':
        sense = 'G'
    else:
        raise NotImplementedError    
    return sense

def get_min_max(op, val, upper_limit):
    if val > upper_limit:
        val = upper_limit
    
    if op == '<=':
        return 1.0, float(val)
    elif op == '==':
        return float(val), float(val)
    elif op == '>=':
        return float(val), float(upper_limit)

class Constraint():
    def __init__(self):
        pass
    
    def add(self, *args, **kwargs):
        pass

class FNRConstraint(Constraint):
    """Limits the FNR to a certain value."""
    def __init__(self, val):
        super().__init__()
        self.val = val
        
    def add(self, names, dataset, binarized_df):
        return {
            'lin_expr': [cplex.SparsePair(ind = [names['lplus']], 
                                            val = [1.0])],
            'senses': ['L'],
            'rhs': [float((dataset.target == 1).sum() * self.val)]
        }
        
    def __str__(self):
        return f"FNR <= {self.val}"

class FPRConstraint(Constraint):
    """Limits the FPR to a certain value."""
    def __init__(self, val):
        super().__init__()
        self.val = val
        
    def add(self, names, dataset, binarized_df):
        return {
            'lin_expr': [cplex.SparsePair(ind = [names['lminus']], 
                                            val = [1.0])],
            'senses': ['L'],
            'rhs': [float((dataset.target == 0).sum() * self.val)]
        }
        
    def __str__(self):
        return f"FPR <= {self.val}"
    
class MaxNumFeatureConstraint(Constraint):
    """Limits the maximum number of items in the checklist."""
    def __init__(self, op, val):
        '''
            op: operator (str), <=, >= or ==
            val: number of items (int)
        '''
        super().__init__()
        self.op = op
        self.val = val
    
    def add(self, names, *args, **kwargs):
        log.warn("Calling add on N_max constraint; should pass to Checklist_MIP.build_problem() instead")
        sense = get_sense(self.op) 
        return {
            'lin_expr': [cplex.SparsePair(ind = [names['N']], 
                                            val = [1.0])],
            'senses': [sense],
            'rhs': [float(self.val)]                                
        }           
        
    def get_min_max(self, n_features):
        return get_min_max(self.op, self.val, n_features)
        
    def __str__(self):
        return f"Total number of selected attributes {self.op} {self.val}."
    
class MConstraint(Constraint): 
    """Limits the number of items to checked off to give a positive prediction."""
    def __init__(self, op, val):
        '''
            op: operator (str), <=, >= or ==
            val: number of items (int)
        '''
        super().__init__()
        self.op = op
        self.val = val
    
    def add(self, names, *args, **kwargs):
        log.warn("Calling add on M_max constraint; should pass to Checklist_MIP.build_problem() instead")
        sense = get_sense(self.op)   
        return {
            'lin_expr':  [cplex.SparsePair(ind = [names['M']], 
                                            val = [1.0])],
            'senses': [sense],
            'rhs': [float(self.val)]                                
        }          
   
    def get_min_max(self, n_features):
        return get_min_max(self.op, self.val, n_features)
    
    def __str__(self):
        return f"M {self.op} {self.val}."        
    
# class LossConstraint(Constraint): 
#     def __init__(self, op = ">=", l = 1):
#         self.l = l
#         self.op = op
        
#     def add(self, names, *args, **kwargs):
#         sense = get_sense(self.op)  
#         return {
#             'lin_expr':   [cplex.SparsePair(ind = [names['l']], 
#                                             val = [1.0])],
#             'senses': [sense],
#             'rhs': [float(self.l)]                                
#         }   
            
#     def __str__(self):
#         return f"Total loss {self.op} {self.l}."
        
class FeaturesPerGroupConstraint(Constraint): 
    '''Select N features from a particular feature group (ex: age)'''
    def __init__(self, feature, op = "<=", N = 1):
        '''
            feature: column name (str), must be in original dataframe
            op: operator (str), <=, >= or ==
            N: number of items (int)
        '''
        self.N = N
        self.feature = feature
        self.op = op
        
    def add(self, names, dataset, binarized_df):
        assert(self.feature in dataset.feature_mapping), f"{self.feature} not in original dataframe!"
        inds, accuracies = [], [] # we use the accuracy of a single feature as the weight for the SOS
        for c, i in enumerate(binarized_df.columns):
            if i in dataset.feature_mapping[self.feature].binarized_names:
                inds.append(c)
                accuracies.append( float((dataset.binarized_df[i] == dataset.target).sum()) )
        sense = get_sense(self.op)  
        if self.N > 1 or self.op != '<=':
            return {
                'lin_expr':[cplex.SparsePair(ind = [names['lamb'][i] for i in inds], 
                                                val = [1.0] * len(inds))],
                'senses': [sense],
                'rhs': [float(self.N)]
            }   
        else:
            return {
                'SOS': cplex.SparsePair(ind = [names['lamb'][i] for i in inds], 
                                                val = np.argsort(np.argsort(accuracies)).astype(float).tolist()),
                'name': f'one_from_{self.feature}'
            } 
             
                        
    def __str__(self):
        return f"Total number of attributes from group '{self.feature}' {self.op} {self.N}."
    
class MostOneofTwoConstraint(Constraint): 
    '''Given names of two binarized features, select at most one of the two'''
    def __init__(self, n1, n2):
        '''
            n1, n2: names of binarized features (str)
        '''
        super().__init__()
        self.n1 = n1
        self.n2 = n2
    
    def add(self, names, dataset, binarized_df):
        try:
            ind1 = list(binarized_df.columns).index(self.n1) 
            ind2 = list(binarized_df.columns).index(self.n2) 
            # return {
            #     'lin_expr': [cplex.SparsePair(ind = [names['lamb'][ind1],names['lamb'][ind2]], 
            #                                 val = [1.0, 1.0])],
            #     'senses': ['L'],
            #     'rhs': [1.0]
            # }
            acc1 = float((dataset.binarized_df[self.n1] == dataset.target).sum())
            acc2 = float((dataset.binarized_df[self.n2] == dataset.target).sum())
            return {
                'SOS': cplex.SparsePair(ind = [names['lamb'][ind1],names['lamb'][ind2]], 
                                             val = [float(acc1 >= acc2), float(acc1 < acc2)]),
                'name': f"at_most_{self.n1}_or_{self.n2}"     
            }
            
        except ValueError: # 1 or more of the features were removed because of duplication
            return None
        
    def __str__(self):
        return f"One of {self.n1}, {self.n2}"      
    
class GroupMetricGapConstraintInterface(Constraint):
    def __init__(self, attr, eps, metric = 'FPR'):
        super().__init__()
        self.attr = attr
        self.eps = eps
        assert metric in ['FPR', 'FNR']
        self.metric = metric

    def add(self, names, dataset, binarized_df):
        assert(self.attr in dataset.protected_attrs)
        grp_vals = dataset.protected_groups[self.attr].unique()
        linexprs = []
        if self.metric == "FPR":
            target_int = -1
        elif self.metric == 'FNR':
            target_int = 1        

        for c, g1 in enumerate(grp_vals):
            m1 = (dataset.protected_groups[self.attr] == g1).values
            n1 = (dataset.target.values == target_int)[m1].sum()
            inds1 = [i for c, i in enumerate(names['z']) if m1[c] and dataset.target.values[c] == target_int]

            for g2 in grp_vals[c+1:]:
                if g1 != g2:
                    m2 = (dataset.protected_groups[self.attr] == g2).values
                    n2 = (dataset.target.values == target_int)[m2].sum()
                    inds2 = [i for c, i in enumerate(names['z']) if m2[c] and dataset.target.values[c] == target_int]                

                    linexprs.append(
                            cplex.SparsePair(ind = inds1 + inds2, 
                                                    val = [1/n1]*len(inds1) + [-1/n2]*len(inds2))
                    )

                    linexprs.append(
                            cplex.SparsePair(ind = inds1 + inds2, 
                                                    val = [-1/n1]*len(inds1) + [1/n2]*len(inds2))
                    )
        
        return {
                'lin_expr': linexprs,
                'senses': ['L'] * len(linexprs),
                'rhs': [float(self.eps)] * len(linexprs)
            } 
    
    def __str__(self):
        return f"|{self.metric}(G1) - {self.metric}(G2)| <= {self.eps} for all (G1, G2) in {self.attr}"


class GroupFPRGapConstraint(GroupMetricGapConstraintInterface):    
    '''Constrains the FPR gap between all groups of an attribute.'''
    def __init__(self, attr, eps):
        '''
            attr: name of attribute to enforce constraint over groups, must appear in dataset.protected_attrs
            eps: size of gap (float), e.g. 0.05
        '''
        super().__init__(attr, eps, 'FPR')

class GroupFNRGapConstraint(GroupMetricGapConstraintInterface):
    '''Constrains the FNR gap between all groups of an attribute.'''
    def __init__(self, attr, eps):
        '''
            attr: name of attribute to enforce constraint over groups, must appear in dataset.protected_attrs
            eps: size of gap (float), e.g. 0.05
        '''
        super().__init__(attr, eps, 'FNR')      

class GroupMetricConstraintInterface(Constraint):
    def __init__(self, attr, eps, metric = 'FNR'):
        super().__init__()
        self.attr = attr
        self.eps = eps
        assert metric in ['FPR', 'FNR']
        self.metric = metric

    def add(self, names, dataset, binarized_df):
        assert(self.attr in dataset.protected_attrs)
        assert(len(names['z']) == len(dataset))
        grp_vals = dataset.protected_groups[self.attr].unique()
        linexprs = []
        if self.metric == "FPR":
            target_int = -1
        elif self.metric == 'FNR':
            target_int = 1  

        for g in grp_vals:
            m = (dataset.protected_groups[self.attr] == g).values            
            n = (dataset.target.values ==  target_int)[m].sum()
            inds = [i for c, i in enumerate(names['z']) if m[c] and dataset.target.values[c] ==  target_int]
            linexprs.append(
                    cplex.SparsePair(ind = inds, 
                                            val = [1/n]*len(inds))
            )
        return {
                'lin_expr': linexprs,
                'senses': ['L'] * len(linexprs),
                'rhs': [float(self.eps)] * len(linexprs)
            } 
    
    def __str__(self):
        return f"{self.metric}(G) <= {self.eps} for all (G) in {self.attr}"

class GroupFNRConstraint(GroupMetricConstraintInterface):
    '''Constrains the worst-case FNR of all groups of an attribute.'''
    def __init__(self, attr, eps):        
        '''
            attr: name of attribute to enforce constraint over groups, must appear in dataset.protected_attrs
            eps: size of gap (float), e.g. 0.2
        '''
        super().__init__(attr, eps, 'FNR')
        
class GroupFPRConstraint(GroupMetricConstraintInterface):
    '''Constrains the worst-case FPR of all groups of an attribute.'''
    def __init__(self, attr, eps):
        '''
            attr: name of attribute to enforce constraint over groups, must appear in dataset.protected_attrs
            eps: size of gap (float), e.g. 0.2
        '''
        super().__init__(attr, eps, 'FPR')

class PredPrevalenceConstraint(Constraint):
    '''Constrains the predicted prevalence of the checklist.'''
    def __init__(self, op, val):
        '''
            op: operator (str), <= or >=
            val: value of constraint (float), e.g. 0.2
        '''
        super().__init__()
        self.op = op
        self.val = val

    def add(self, names, dataset, binarized_df):
        sense = get_sense(self.op) 
        n_plus = float((dataset.target == 1).sum())
        return {
            'lin_expr': [cplex.SparsePair(ind = [names['lminus'], names['lplus']],
                                          val = [1.0, -1.0])],
            'senses': [sense],
            'rhs': [float(len(dataset.target) * self.val - n_plus)]
        }

    def __str__(self):
        return f"Predicted Prevalence <= {self.val}"

class ItemSelectionConstraint(Constraint): 
    '''Constraints the checklist to use {<=, ==, >=} K items from a provided list of binary feature names.'''
    def __init__(self, feat_names, op, K):
        '''
            feat_names: list of binary features in binary dataset
            op: operator (str), <=, >= or ==
            K: number of items (int)
        '''
        super().__init__()
        self.feat_names = feat_names
        self.op = op
        self.K = K
    
    def add(self, names, dataset, binarized_df):
        inds = []
        for c, i in enumerate(binarized_df.columns): # disregard features not in binarized_df
            if i in self.feat_names:
                inds.append(c)
        sense = get_sense(self.op)  
        return {
            'lin_expr':[cplex.SparsePair(ind = [names['lamb'][i] for i in inds], 
                                            val = [1.0] * len(inds))],
            'senses': [sense],
            'rhs': [float(self.K)]
        }   
                        
    def __str__(self):
        if len(self.feat_names) <= 5:
            return f"Use {self.op} {self.K} items from {self.feat_names}"
        return  f"Use {self.op} {self.K} items from {self.feat_names[:5]}"[:-1] + ', ...'