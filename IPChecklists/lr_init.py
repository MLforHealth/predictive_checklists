from IPChecklists.Checklist import Checklist
import numpy as np
from pathlib import Path
import pickle
import logging
from sklearn.pipeline import Pipeline
logging.basicConfig(level='INFO')
log = logging.getLogger()

def checklists_from_lr(lr_model, train_ds, test_ds = None, use_basic = False):
    """Generates a list of checklists from a trained L1 regularized LR model.

        Args:
            lr_model: logistic regression model; sklearn Pipeline or LogisticRegression
            train_ds: training dataset
            test_ds: test dataset
            use_basic: whether to use a more basic version of UnitWeighting that does not take feature groups into account

        Returns:
            checklists: list of Checklist objects
            train_metrics: dictionary of training set metrics
            test_metrics: dictionary of test set metrics
            N: number of items of all checklists
    """
    if not train_ds.add_complements:
        log.warn("Dataset does not contain complements. Unit weighting might give poor performance " +
                "under the presence of negative parameters.")
    if isinstance(lr_model, Pipeline):
        coefs = lr_model.steps[-1][1].coef_[0]
    else:
        coefs = lr_model.coef_[0]
    weights = np.copy(coefs)
    abs_weights = np.abs(weights)
    
    if not use_basic:
        for i in train_ds.feature_mapping: # set all features in one group that do not have largest params to zero
            inds = []
            for j in train_ds.feature_mapping[i].binarized_names:
                try:
                    inds.append(list(train_ds.binarized_df.columns).index(j))
                except ValueError:
                    pass
            if len(inds) == 0:
                continue
            max_abs = np.max(abs_weights[np.array(inds)])
            for j in inds:
                if np.abs(weights[j]) < max_abs:
                    weights[j] = 0
     
    weights[weights > 0] = 1
    # if a weight is negative, take the complement feature as 1
    for i in train_ds.feature_mapping:
        for j in train_ds.feature_mapping[i].binarized_names:
            try:
                cur_ind = list(train_ds.binarized_df.columns).index(j)
            except ValueError:
                continue
            if weights[cur_ind] < 0 and train_ds.add_complements:         
                try:
                    oppo_names = [k for k in train_ds.feature_mapping[i].complement_pairs if j in k][0]
                    oppo_name = oppo_names[1 - oppo_names.index(j)]
                    oppo_ind = list(train_ds.binarized_df.columns).index(oppo_name)
                    weights[oppo_ind] = 1
                    weights[cur_ind] = 0    
                except (ValueError, IndexError) as e: # complement feature was dropped from df due to redundancy, or never existed
                    log.warn(f"Complement feature to {j} not found.")
                    weights[cur_ind] = 0    
                
    weights[weights < 0] = 0
    N = (weights == 1).sum()     
    checklists = []
    train_metrics, test_metrics = [], []
    for M in range(1, N+1):
        checklists.append(Checklist(from_mip = False, from_weights = True, lamb = weights,
                                         M = M, column_names = list(train_ds.binarized_df.columns)))
        train_metrics.append(checklists[-1].get_metrics(train_ds))
        if test_ds is not None:
            test_metrics.append(checklists[-1].get_metrics(test_ds))
        
    return checklists, train_metrics, test_metrics, N
