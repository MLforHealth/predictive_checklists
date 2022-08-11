import numpy as np
import pandas as pd
from .utils import infer_types, set_seed_wrapper
import logging
import optbinning
import copy

logging.basicConfig(level='INFO')
log = logging.getLogger()

class Feature():
    def __init__(self, overall_name, binarized_names, dtype, complement_pairs = []):
        self.overall_name = overall_name
        self.binarized_names = binarized_names
        self.type = dtype
        self.count = len(binarized_names)
        self.complement_pairs = complement_pairs

def compr(x, val, op):
    if op == '>=':
        return x >= val
    elif op == '>':
        return x > val
    elif op == '<=':
        return x <= val
    elif op == '<':
        return x < val
    elif op == '==':
        return x == val
    else:
        raise NotImplementedError(op)

inverse_ops = {
     '==': '!=',
     '>=': '<',
    '<=': '>',
    '!=': '=='
}

def compute_optbin_thresholds(df, dtypes, target, max_n_bins = 5):
    target = np.copy(target)
    target[target == -1] = 0
    thresholds = {}
    for col in dtypes:
        optb = set_seed_wrapper(optbinning.OptimalBinning)(name = col, dtype = dtypes[col] if dtypes[col] == 'categorical' else 'numerical',
                                         solver="cp", max_n_bins = max_n_bins if dtypes[col] == 'numeric' else None)
        optb.fit(df[col].values, target)
        thresholds[col] = list(optb.splits)

        if dtypes[col] == 'categorical':
            thresholds[col] = [j for i in thresholds[col] for j in i]

    return thresholds


class BinaryDataset():
    '''
    Given a dataframe of features and a single target, binarizes the features
        and prepares it for checklist training

    Potentially useful attributes:
       - binarized_df (Dataframe)
       - target (array)
       - col_types (dictionary mapping column to data type)
       - thresholds (dictionary mapping column to list of floats)
       - feature_mapping (dictionary mapping column to a Feature object, which keeps track of binarized features and complement pairs)
       - remaining_features (list of strings containing names of binarized features)
    '''
    def __init__(self, df, target_name, pos_label, col_subset = None, percentile_inc = 25, op = '>=',
                    already_binarized = False, thresholds = None, add_complements = True, use_optbin = False,
                 max_n_bins = 5, complement_includes_missing = True, use_missing_features = True,
                 protected_attrs = [], categorical_bin = 'all'):
        '''
        Args:
            df (Dataframe): dataframe with features for modelling, and `target_name` as columns
            target_name (str): string of column name to use as target; must appear as a column in df
            pos_label (str/int): string/int of label to use as the positive label (i.e. what the checklist predicts), e.g. 1.0
            col_subset (list[str]): if df contains redundant columns, col_subset is the subset of columns to use as features
            percentile_inc (int/float): for numeric variables, the percentile increment to binarize by
            op (char/dict[str:char]): for ordinal and numeric variables, the direction of binarized features
            already_binarized: features and target in df are pre-binarized
            thresholds (dict[str:list]): manually provide thresholds for each feature
            add_complement (bool/dict[str:bool]): adds the complement of all generated features
            use_optbin (bool): whether to compute thresholds using the optbinning package, or by using percentile thresholds
            max_n_bins (int): maximum number of bins for continuous features when computing thresholds with optbinning
            complement_includes_missing (bool): whether the complement of a column will be positive if the feature is missing
            use_missing_features (bool): whether to include whether a particular feature is missing as a variable
            protected_attrs (list[str]): names of columns to use as protected attributes. Must appear as columns in df. 
            categorical_bin (str): one of ['all', 'one']. If all, will use each unique value as thresholds for categorical variables. Otherwise, will use most frequent bin.
        '''
        # set fields
        assert(target_name in df)        
        assert(categorical_bin in ['all', 'one'])
        self.target_name = target_name
        self.pos_label = pos_label
        self.col_subset = col_subset
        self.df = df if col_subset is None else df[[target_name] + col_subset]
        self.percentile_inc = percentile_inc
        self.op = op
        self.already_binarized = already_binarized
        self.thresholds = thresholds
        self.add_complements = add_complements
        self.use_optbin = use_optbin
        self.max_n_bins = max_n_bins
        self.complement_includes_missing = complement_includes_missing
        self.use_missing_features = use_missing_features     
        self.protected_attrs = protected_attrs
        self.categorical_bin = categorical_bin
                
        if protected_attrs:
            self.protected_groups = df[protected_attrs]        

        if self.already_binarized:
            self.target = self._binarize_target(self.df[self.target_name])

            # check df
            self.binarized_df = self.df.drop(columns = [target_name])
            for i in self.binarized_df:
                if i != target_name:
                    assert ((~self.binarized_df[i].isin([0, 1])).sum()) == 0, f'{i} not binary!'

            # initialize other required fields
            self.col_types = {i: 'prebinarized' for i in self.binarized_df}
            self.thresholds = None
            self.feature_mapping = {i: Feature(i, [i], 'prebinarized') for i in self.binarized_df}

        else:
            # binarizes target
            self.target = self._binarize_target(self.df[self.target_name])

            # binarizes df
            self._binarize_df()

        # remove columns with the same data
        before_removal_columns = self.binarized_df.columns
        self.remaining_features = list(self.binarized_df.T.drop_duplicates().T.columns)

        # remove columns where all samples are the same value
        i=0
        while i < len(self.remaining_features):
            if len(self.binarized_df[self.remaining_features[i]].unique()) == 1:
                del self.remaining_features[i]
            else:
                i += 1

        self.binarized_df = self.binarized_df[self.remaining_features]
        log.info("Removed %s non-informative columns: %s" % (len(before_removal_columns) - len(self.remaining_features),
                                                                set(before_removal_columns) - set(self.remaining_features)))
        log.info(f"Binary dataframe: {self.binarized_df.shape[1]} binary features and {self.binarized_df.shape[0]} samples")

    def _binarize_target(self, target_arr):
        assert(len(target_arr.unique()) == 2), "More than two unique target values!"
        assert(self.pos_label in target_arr.unique()), "Positive label not present in target column!"
        target = (target_arr == self.pos_label).astype(int)
        target[target == 0] = -1
        return target

    def _binarize_df(self):
        df = self.df.copy()
        df = df.drop(columns = [self.target_name])
        self.pctle_list = np.arange(self.percentile_inc, 100, self.percentile_inc)
        self.col_types = infer_types(df)
        if self.thresholds is None:
            if self.use_optbin:
                self.thresholds = compute_optbin_thresholds(df, self.col_types, self.target, self.max_n_bins)
            else: # build thresholds from percentiles
                self.thresholds = {}                
                for col in df:
                    binarized_names = []
                    if self.col_types[col] == 'categorical':
                        if self.categorical_bin == 'all':
                            self.thresholds[col] = df[col].unique()
                        elif self.categorical_bin == 'one':
                            self.thresholds[col] = [df[col].value_counts().sort_values(ascending = False).index[0]]
                        if len(self.thresholds[col]) > df.shape[0] // 2:
                            log.warning(f'categorical column "{col}" has many ({len(self.thresholds[col])}) unique values')
                    else:
                        if self.col_types[col] == 'ordinal':
                            if self.categorical_bin == 'all':
                                split_vals = np.sort(df[col].unique())
                            elif self.categorical_bin == 'one':
                                split_vals = [df[col].dropna().median()]
                        elif self.col_types[col] == 'numeric':
                            split_vals = np.sort(pd.unique(np.nanpercentile(df[col].astype(float).values, q = self.pctle_list)))
                            split_vals = split_vals[~np.isnan(split_vals)].tolist()
                            if pd.isnull(df[col]).sum() > 0:
                                split_vals.append(np.nan)
                        else:
                            raise NotImplementedError

                        self.thresholds[col] = split_vals

        df, self.feature_mapping = self._apply_existing_thresholds(df)
        self.binarized_df = df.drop(columns = list(self.col_types.keys()))

    def _apply_existing_thresholds(self, df):
        feature_mapping = {}
        for col in df:
            binarized_names = []
            complement_pairs = []
            add_complements_i = self.add_complements if isinstance(self.add_complements, bool) else self.add_complements[col]
            op_i = self.op if isinstance(self.op, str) else self.op[col]
            if self.col_types[col] == 'categorical':
                for val in self.thresholds[col]:
                    if pd.isnull(val):
                        if not self.use_missing_features:
                            continue
                        name = col+"_Missing"
                        df[name] = pd.isnull(df[col]).astype(int)
                        comp_name = col+"_NotMissing"
                    else:
                        name = f'{col}=={val}'
                        df[name] = (df[col] == val).astype(int)
                        comp_name = f'{col}!={val}'
                    binarized_names.append(name)

                    if add_complements_i:
                        binarized_names.append(comp_name)
                        
                        if self.complement_includes_missing:                        
                            df[comp_name]= (df[name] == 0).astype(int)
                        else:
                            df[comp_name] = ((df[col] != val) & (~pd.isnull(df[col]))).astype(int)
                            
                        complement_pairs.append((name, comp_name))

                # only 2 categories, can potentially use other category as compl
                if add_complements_i and len(self.thresholds[col]) == 2:
                    n1, c1, n2, c2 = binarized_names[-4:]
                    if (df[n1] == df[c2]).all():
                        complement_pairs = complement_pairs[:-2]
                        complement_pairs.append((n1, n2))
                        binarized_names = binarized_names[:-4] + [n1, n2]
                        df = df.drop(columns = [c1, c2])

            else:
                split_vals = self.thresholds[col]
                for c, val in enumerate(split_vals):
                    if pd.isnull(val):
                        if not self.use_missing_features:
                            continue
                        name = col+"_Missing"
                        df[name] = pd.isnull(df[col]).astype(int)
                        comp_name = col+"_NotMissing"
                    else:
                        name = f'{col}{op_i}{val}'
                        comp_name = f'{col}{inverse_ops[op_i]}{"~" if self.complement_includes_missing else ""}{val}'
                        df[name] = compr(df[col], val, op_i).fillna(0).astype(int)

                    binarized_names.append(name)
                    if add_complements_i:
                        binarized_names.append(comp_name)
                        if self.complement_includes_missing:   
                            df[comp_name]= (df[name] == 0).astype(int)
                        else:
                            df[comp_name]= compr(df[col], val, inverse_ops[op_i]).fillna(0).astype(int)
                        complement_pairs.append((name, comp_name))

            feature_mapping[col] = Feature(col, binarized_names, dtype = self.col_types[col], complement_pairs = complement_pairs)

        return df, feature_mapping

    def drop_binary_features(self, feat_list):
        '''Drops a list of binary features in-place from the binarized dataframe. Should be used before creating the model.'''
        for i in feat_list:
            assert i in self.binarized_df, f'%"{i}" not in binary feature name'
            self.remaining_features.remove(i)
        
        self.binarized_df = self.binarized_df.drop(columns = feat_list)

    def subset_by_protected_group(self, attribute_name, grp_name):
        mask = self.protected_groups[attribute_name] == grp_name
        return self.binarized_df[mask], self.target[mask]    

    def shuffle_with_replacement(self, stratify_by = None):
        if stratify_by is None:
            idxs = np.random.choice(np.arange(len(self.binarized_df)), size = len(self.binarized_df))
        else:
            protected_groups_with_idx = self.protected_groups[[stratify_by]].assign(idx = np.arange(len(self.protected_groups)))
            idxs = protected_groups_with_idx.groupby(stratify_by).apply(lambda x: x.sample(n = len(x), replace = True)).sample(frac=1)['idx'].values
        
        new_ds = copy.deepcopy(self)
        new_ds.df = self.df.iloc[idxs]
        new_ds.binarized_df = self.binarized_df.iloc[idxs]
        new_ds.target = self.target.iloc[idxs]
        new_ds.protected_groups = self.protected_groups.iloc[idxs]
        return new_ds

    def apply_transform(self, df_test):    
        '''Transforms a new dataframe using the thresholds that have been computed on this dataset.'''
        if self.already_binarized:
            # log.warn("apply_transform called with already_binarized = True")
            target = self._binarize_target(df_test[self.target_name])
            df_new = df_test.drop(columns = [self.target_name])
        else:
            df_new = df_test.copy()
            df_new = df_new[self.df.columns]

            # binarize target
            target = self._binarize_target(df_new[self.target_name])

            # apply threshold to df
            df_new = df_new.drop(columns = [self.target_name])
            df_new, _ = self._apply_existing_thresholds(df_new)

            df_new = df_new.drop(columns = list(self.col_types.keys()))

        # create returned BinaryDataset
        new_ds = BinaryDataset.__new__(BinaryDataset)
        new_ds.df = df_test
        new_ds.target_name = self.target_name
        new_ds.pos_label = self.pos_label
        new_ds.col_subset = self.col_subset
        new_ds.percentile_inc = self.percentile_inc
        new_ds.op = self.op
        new_ds.already_binarized = self.already_binarized
        new_ds.binarized_df = df_new[self.remaining_features]
        new_ds.target = target
        new_ds.col_types = self.col_types
        new_ds.thresholds = self.thresholds
        new_ds.feature_mapping = self.feature_mapping
        new_ds.add_complements = self.add_complements
        new_ds.complement_includes_missing = self.complement_includes_missing
        new_ds.remaining_features = self.remaining_features
        new_ds.use_missing_features = self.use_missing_features
        new_ds.protected_attrs = self.protected_attrs
        new_ds.categorical_bin = self.categorical_bin
        
        if self.protected_attrs:
            new_ds.protected_groups = df_test[self.protected_attrs]   

        return new_ds

    def __len__(self):
        return self.binarized_df.shape[0]

