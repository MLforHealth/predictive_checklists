from collections import defaultdict
from IPChecklists.utils import random_supersample, probability_metrics, convert_target_domain, binary_metrics
import pandas as pd
import numpy as np
from pathlib import Path
from IPChecklists.dataset import BinaryDataset
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import LabelEncoder
from IPChecklists.utils import split_folds

all_datasets = ['CRRT', 'Heart', 'CShock_eICU', 'CShock_MIMIC', 
            'MIMICMortality', 'PTSD', 'ADHD', 'Readmit'] 

data_dir = Path(__file__).absolute().parent.parent/'data/' # update this

class Data():
    def __init__(self, use_optbin, percentile_inc = 20, max_n_bins = 5, balance_train = False, balance_test = False, use_naive = False, fold = 0):
        if use_naive:
            percentile_inc = 50
        self.df_train, self.df_test, self.train_ds, self.test_ds, self.all_cols = self._load_data(use_optbin, percentile_inc, max_n_bins, 
                            balance_train = balance_train, balance_test = balance_test, use_naive = use_naive, fold = fold)

    def get_data(self):
        return self.df_train, self.df_test, self.train_ds, self.test_ds, self.all_cols

    def summarize(self):
        df = pd.concat((self.df_train, self.df_test))        
        target_name = self.train_ds.target_name
        pos_label = self.train_ds.pos_label

        res = {
            'n_samples': df.shape[0],
            'n_positive': (df[target_name] == pos_label).sum(),
            'prebinarized': self.prebinarized
        }
        res['% positive'] = res['n_positive']/res['n_samples']
        res['n_orig_features'] = len(self.all_cols)
        res['n_bin_features'] = self.train_ds.binarized_df.shape[1]

        return res

    def train_sklearn_model(self, model_type = 'xgb', dtype = 'binarized', include_fairness = False, fairness_attrs = None):  
        '''
        Trains a simple sklearn model on either the original or binarized data. Grid searches simple hyperparameters
        Inputs:
            model_type: {'xgb', 'lr'}
            dtype: {'binarized', 'original'}
        ''' 
        assert (model_type == 'xgb' or dtype == 'binarized')
        if model_type == 'xgb':     
            estimator = XGBClassifier(random_state = 42,  eval_metric = 'error')
            grid = {
                'max_depth': range(1, 8) 
            }
        elif model_type == 'lr':
            estimator = LogisticRegression(penalty = 'l1', random_state = 42, solver = 'liblinear')
            grid = {
                'C': 10**np.linspace(-4, 1, 20) 
            }
        else:
            raise NotImplementedError
        gs = GridSearchCV(estimator = estimator, param_grid = grid, scoring = 'accuracy', 
                    n_jobs = -1, refit = True, cv = 5)

        if dtype == 'original':
            df_train, df_test = self.df_train.copy(), self.df_test.copy()
            les = {}
            for col, dtype in self.train_ds.col_types.items():
                if dtype == 'categorical':
                    les[col] = LabelEncoder().fit(df_train[col].fillna("Missing"))
                else:
                    df_train[col] = df_train[col].astype(float)
                    df_test[col] = df_test[col].astype(float)
            for col in les:
                df_train[col] = les[col].transform(df_train[col].fillna("Missing"))
                df_test[col] = les[col].transform(df_test[col].fillna("Missing"))

            les['target'] = LabelEncoder().fit(df_train[self.train_ds.target_name])
            df_train[self.train_ds.target_name] = les['target'].transform(df_train[self.train_ds.target_name])
            df_test[self.test_ds.target_name] = les['target'].transform(df_test[self.test_ds.target_name])

            X_train, X_test = df_train[self.all_cols].values, df_test[self.all_cols].values
            y_train, y_test = df_train[self.train_ds.target_name].values, df_test[self.test_ds.target_name].values

        elif dtype == 'binarized':
            X_train, X_test = self.train_ds.binarized_df.values, self.test_ds.binarized_df.values
            y_train, y_test = convert_target_domain(self.train_ds.target), convert_target_domain(self.test_ds.target)
        else:
            raise NotImplementedError

        gs = gs.fit(X_train, y_train)
        model = gs.best_estimator_
        preds_train = model.predict_proba(X_train)[:, 1]
        preds_test = model.predict_proba(X_test)[:, 1]

        metrics = {}
        metrics['prob_metrics_train'] = probability_metrics(y_train, preds_train)
        metrics['prob_metrics_test'] = probability_metrics(y_test, preds_test)
        metrics['bin_metrics_train'] = binary_metrics(y_train, preds_train >= 0.5)
        metrics['bin_metrics_test'] = binary_metrics(y_test, preds_test >= 0.5)
        return model, metrics

class CRRT(Data):
    prebinarized = False
    def _load_data(self, use_optbin, percentile_inc = 20, max_n_bins = 5, balance_train = False, balance_test = False, use_naive = False, fold = 0): 
        df = pd.concat([pd.read_pickle(data_dir/'crrt_train.pkl'),
                    pd.read_pickle(data_dir/'crrt_test.pkl')], ignore_index = True).reset_index(drop = True)
        df_train, df_test = split_folds(df, 'target', fold)

        df_train.loc[~df_train.ethnicity.isin(['Caucasian', 'Black']), 'ethnicity'] = 'Other'
        df_test.loc[~df_test.ethnicity.isin(['Caucasian', 'Black']), 'ethnicity'] = 'Other'

        df_train['ethnicity_gender'] = df_train['ethnicity'] + '_' + df_train['gender']
        df_test['ethnicity_gender'] = df_test['ethnicity'] + '_' + df_test['gender']

        if balance_train:
            df_train = random_supersample(df_train, 'target')
        if balance_test:
            df_test = random_supersample(df_test, 'target')

        cont_cols = ['Sodium', 'Potassium', 'Chloride', 'Bicarbonate', 'Creatinine', 'AST', 'ALT', 'Platelets',
                    'RDW', 'norepi', 'timesinceicu', 'anchor_age', 'BloodpH', 'MCV']
                    # 'pre_input', 'pre_output', 'net_pre_input']

        cat_cols = ['gender', 'pvd']
        ord_cols = ['sofa_admit']

        all_cols = cont_cols + cat_cols + ord_cols

        directions = {
            'Potassium': '>=',
            'Chloride': '>=',
            'Bicarbonate': '<=',
            'Creatinine': '>=',
            'AST': '>=',
            'ALT': '>=',
            'Platelets': '<=',
            'RDW': '>=',
            'BloodpH': '<=',
            'sofa_admit': '>='
        }

        add_complements = {i: False if i in directions else True for i in all_cols}
        for i in all_cols:
            if i not in directions:
                directions[i] = '>='

        train_ds = BinaryDataset(df_train, target_name = 'target', pos_label = 1, col_subset = cont_cols + cat_cols + ord_cols,
                                thresholds = None, add_complements = add_complements if not use_naive else True,
                                use_optbin = use_optbin, percentile_inc = percentile_inc,
                                max_n_bins = max_n_bins, op = directions,
                                use_missing_features = False, complement_includes_missing = False,
                                protected_attrs = ['gender', 'ethnicity', 'age_bin', 'ethnicity_gender'],
                                categorical_bin = 'one' if use_naive else 'all')

        test_ds = train_ds.apply_transform(df_test)

        return df_train, df_test, train_ds, test_ds, all_cols

class Heart(Data):
    prebinarized = False
    def _load_data(self, use_optbin, percentile_inc = 20, max_n_bins = 5, balance_train = False, balance_test = False, use_naive = False, fold = 0):
        df = pd.read_csv(data_dir/"heart.csv")
        cont_cols = ['trestbps', 'chol', 'thalach', 'age', 'oldpeak']
        for i in cont_cols:
            df[i] = df[i].astype(float)
            
        cat_cols = ['cp', 'thal', 'ca', 'slope', 'restecg']
        for i in cat_cols:
            df[i] = df[i].astype(str)

        df_train, df_test= split_folds(df, 'target', fold)     
        target = 'target'
        pos_label = 1

        if balance_train:
            df_train = random_supersample(df_train, target)
        if balance_test:
            df_test = random_supersample(df_test, target)

        train_ds = BinaryDataset(df_train, target_name = target, pos_label = pos_label, use_optbin = use_optbin, percentile_inc = percentile_inc,
                                max_n_bins = max_n_bins, add_complements = True, categorical_bin = 'one' if use_naive else 'all')
        test_ds = train_ds.apply_transform(df_test)

        return df_train, df_test, train_ds, test_ds, [i for i in df_train.columns if i != target]  

class CShockBase(Data):
    prebinarized = True
    def __init__(self, dataset_name, *args, **kwargs):        
        if dataset_name == 'eICU':
            self.xvar, self.yvar = 'X', 'Y'
        elif dataset_name == 'MIMIC':
            self.xvar, self.yvar = 'X_test', 'Y_test'
        else:
            raise NotImplementedError

        super().__init__(*args, **kwargs)

    def _load_data(self, *args, **kwargs):
        raw = pd.read_pickle(data_dir/"cshock20_processed.pickle")
        df = (pd.DataFrame(raw['data'][self.xvar], columns = raw['data']['variable_names']) # eICU
                .drop(columns = ['(Intercept)']))
        df['target'] = raw['data'][self.yvar]

        if kwargs['use_naive']:
            df = df.drop(columns = ['age_geq_60', 'age_geq_80', 'hr_min_geq_100', 'hr_max_leq_60', 
                'sys_bp_min_lt_80', 'sys_bp_max_leq_80', 'resp_rate_max_leq_12', 'resp_rate_min_geq_25', 'sp_o2_min_geq_95',
                'hemoglobin_min_leq_8', 'platelet_min_leq_150', 'wbc_max_leq_11'])

        target = 'target'

        train, test = split_folds(df, target, kwargs['fold']) 

        if kwargs['balance_train']:
            train = random_supersample(train, target)
        if kwargs['balance_test']:
            test = random_supersample(test, target)

        train_ds = BinaryDataset(train, target_name = target, pos_label = 1, already_binarized = True)
        test_ds = train_ds.apply_transform(test)

        return train, test, train_ds, test_ds, [i for i in train.columns if i != target]  

class CShock_MIMIC(CShockBase):
    def __init__(self, *args, **kwargs):
        super().__init__('MIMIC', *args, **kwargs)
   
class CShock_eICU(CShockBase):
    def __init__(self, *args, **kwargs):
        super().__init__('eICU', *args, **kwargs)

class MIMICMortality(Data):
    prebinarized = False
    def _load_data(self, use_optbin, percentile_inc = 20, max_n_bins = 5, balance_train = False, balance_test = False, use_naive = False, fold = 0):
        df = pd.concat([pd.read_pickle(data_dir/'mort_train.pkl')
                , pd.read_pickle(data_dir/'mort_test.pkl')], ignore_index = True).reset_index(drop = True)

        df['Age'] = df['Age'].astype(float)
        target = 'target'
        df_train, df_test = split_folds(df, target, fold)

        feature_names = [i for i in df_train.columns if i.endswith('count') or i.endswith('mean') or i in ['Age', 'Gender']]

        if balance_train:
            df_train = random_supersample(df_train, target)
        if balance_test:
            df_test = random_supersample(df_test, target)

        train_ds = BinaryDataset(df_train, target_name = target, pos_label = 1, col_subset = feature_names,
                                 use_optbin = use_optbin, percentile_inc = percentile_inc,
                                max_n_bins = max_n_bins, add_complements = True,
                                categorical_bin = 'one' if use_naive else 'all')
        test_ds = train_ds.apply_transform(df_test)

        return df_train, df_test, train_ds, test_ds, feature_names

class AuroraBase(Data):
    prebinarized = False
    def __init__(self, dataset_name, target_name, *args, **kwargs):        
        self.dataset_name = dataset_name
        self.df_path = data_dir/f"aurora_{dataset_name}_df.pkl"
        self.meta_path = data_dir/f"aurora_{dataset_name}_meta.pkl"
        self.target_name = target_name
        super().__init__(*args, **kwargs)

    def _load_data(self, use_optbin, percentile_inc = 20, max_n_bins = 5, balance_train = False, balance_test = False, use_naive = False, fold = 0):
        df = pd.read_pickle(self.df_path)
        meta = pickle.load(self.meta_path.open('rb'))
        feature_names = list(meta.keys())
        df['race_sex'] = df['race'] + '_' + df['sex']

        target = self.target_name
        train, test = split_folds(df, target, fold)

        if balance_train:
            train = random_supersample(train, target)
        if balance_test:
            test = random_supersample(test, target)

        train_ds = BinaryDataset(train, target_name = target, pos_label = 1, col_subset = feature_names,
                                 use_optbin = use_optbin, percentile_inc = percentile_inc,
                                max_n_bins = max_n_bins, add_complements = False, protected_attrs = ['sex','race','age_bin', 'race_sex'],
                                categorical_bin = 'one' if use_naive else 'all')
        test_ds = train_ds.apply_transform(test)

        return train, test, train_ds, test_ds, feature_names

class PTSD(AuroraBase): # rule-based diagnosis
     def __init__(self, *args, **kwargs):
         super().__init__('ptsd', 'ptsd_label2', *args, **kwargs)

class ADHD(AuroraBase):
     def __init__(self, *args, **kwargs):
         super().__init__('adhd', 'adhd_label', *args, **kwargs)

class Readmit(Data):
    prebinarized = False
    def _load_data(self, use_optbin, percentile_inc = 20, max_n_bins = 5, balance_train = False, balance_test = False, use_naive = False, fold = 0):
        df = pd.read_excel(data_dir/'MGH_DataForMIT.xlsx', engine = 'openpyxl').dropna(subset = ['Readmit'])

        feature_names = [i for i in df.columns if i not in ['ID', 'Readmit']]
        target = 'Readmit'
        df_train, df_test = split_folds(df, target, fold)

        if balance_train:
            df_train = random_supersample(df_train, target)
        if balance_test:
            df_test = random_supersample(df_test, target)

        train_ds = BinaryDataset(df_train, target_name = target, pos_label = 1, col_subset = feature_names,
                                 use_optbin = use_optbin, percentile_inc = percentile_inc,
                                max_n_bins = max_n_bins, add_complements = True,
                                categorical_bin = 'one' if use_naive else 'all')
        test_ds = train_ds.apply_transform(df_test)

        return df_train, df_test, train_ds, test_ds, feature_names

def get_dataset(dataset):
    if dataset not in globals():
        raise NotImplementedError(dataset)
    return globals()[dataset]
