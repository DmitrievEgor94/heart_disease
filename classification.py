#!/usr/bin/env python
# coding: utf-8
# ## Задача по прогнозированию болезни сердца https://archive.ics.uci.edu/ml/datasets/heart+Disease
# -----------------------------------------------

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------

import pandas as pd
import seaborn as sns
sns.set(font_scale=1.3)
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import numpy as np
from warnings import simplefilter
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from category_encoders.cat_boost import CatBoostEncoder
simplefilter(action='ignore', category=FutureWarning)
import xgboost as xgb
from xgboost import XGBClassifier

# -----------------------------------------------

SEED = 0

# -----------------------------------------------

df = pd.read_csv('heart.dat',
                 names=[
                     'age', 'sex', 'chess_pain_type', 'rg_blood_pressure',
                     'sr_cholestoral', 'fg_blood_sugar', 'cardio',
                     'max_heart_rate', 'exercise_ind_angina', 'oldpeak',
                     'slpe_peak_exercize', 'ves_number', 'thal', 'ill'
                 ],
                 header=None,
                 sep=' ',
                 index_col=False)

# -----------------------------------------------

df.head()

# ## EDA
# -----------------------------------------------

df.describe()

# -----------------------------------------------

df['ill'] -= 1

# -----------------------------------------------

df.nunique()

# -----------------------------------------------

num_features = [
    'age', 'rg_blood_pressure', 'sr_cholestoral', 'max_heart_rate', 'oldpeak'
]
cat_features = [
    feature for feature in df.columns if feature not in num_features + ['ill']
]

# -----------------------------------------------

df.dtypes

# -----------------------------------------------

fig, axs = plt.subplots(len(num_features), figsize=(10, 30))
axs = axs.flatten()
for i, item in enumerate(num_features):
    df[df.ill == 0][item].plot.kde(title=item, ax=axs[i], label='healthy')
    df[df.ill == 1][item].plot.kde(title=item, ax=axs[i], label='ill')
    axs[i].legend(fontsize=15)

# -----------------------------------------------

for i, item in enumerate(cat_features):
    sns.catplot(kind='count', data=df, x=item, hue='ill')

# -----------------------------------------------

check_for_use_f = ['rg_blood_pressure']

# -----------------------------------------------

train_features = num_features + cat_features

# -----------------------------------------------

buf = df['slpe_peak_exercize'] == 2
df.loc[df['slpe_peak_exercize'] == 3, 'slpe_peak_exercize'] = 2
df.loc[buf, 'slpe_peak_exercize'] = 3

# -----------------------------------------------

sns.catplot(kind='count', data=df, x='slpe_peak_exercize', hue='ill')

# ## Search best model
# -----------------------------------------------

log_reg_ftrs = {
    'features': {
        'C':  hp.loguniform('x_C', -10, 1),
        'solver': 'liblinear',
        'tol': hp.loguniform('x_tol', -13, -1),
        'penalty': 'l2'
    },
    'model': LogisticRegression
}
random_forest_ftrs = {
    'features' : {
        'max_depth': hp.choice('max_depth', range(1, 10)),
        'max_features': hp.choice('max_features', range(1,10)),
        'n_estimators': hp.choice('n_estimators', range(10,100)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'min_samples_split': hp.choice('min_samples_split', range(2, 20)),
        'min_samples_leaf': hp.choice('min_samples_leaf', range(2, 20)),
        'random_state': SEED
    },
    'model': RandomForestClassifier
}
    
xgb_ftrs = {
    'features' : {
        'n_estimators': hp.choice('n_estimators', np.arange(3, 150, 5)),
        'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
        'max_depth':  hp.choice('max_depth', np.arange(1, 10, dtype=int)),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0, 10, 0.5),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'objective': hp.choice('objective', ('reg:gamma', 'reg:squarederror')),
        'seed':SEED
    },
    'model': XGBClassifier
}    

# -----------------------------------------------

hp_params = {
    'n_fold': 5,
    'metric': roc_auc_score
}

# -----------------------------------------------

def get_model_est(model, metric, X, y, folds = 5, params={}):  
    return cross_val_score(model,
                     X,
                     y,
                     cv=KFold(folds, shuffle=True, random_state=SEED),
                     scoring=make_scorer(metric)).mean()

# -----------------------------------------------

def get_model_est_hyper_opt(params):
    model_cnstr = params['model_and_ftrs_space']['model']
    ftrs = params['model_and_ftrs_space']['features']
    metric = params['metric']
    n_fold = params['n_fold']
        
    model = model_cnstr(**ftrs)
        
    return -cross_val_score(model,
                           X,
                           y,
                           cv=KFold(n_fold, shuffle=True, random_state=SEED),
                           scoring=make_scorer(metric)).mean()

# -----------------------------------------------

def get_model_est_features_enc(model, metric, X, y, encoder, cat_features):
    k_fold = KFold(5, shuffle=True, random_state=SEED)
    metrics = []
    for train_ind, test_ind in k_fold.split(X, y):
        X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
        y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
        enc = encoder(cols=cat_features)
        X_train = enc.fit_transform(X_train, y_train)
        model.fit(X_train, y_train)
        metrics.append(metric(y_test, model.predict(enc.transform(X_test))))
    return np.array(metrics).mean()

# -----------------------------------------------

X = df[train_features]
y = df['ill']

# ### Random Forest
# -----------------------------------------------

hp_params['model_and_ftrs_space'] = random_forest_ftrs

# -----------------------------------------------

bst_prms_rndm_frst = space_eval(hp_params['model_and_ftrs_space']['features'],fmin(fn=get_model_est_hyper_opt, space=hp_params, algo=tpe.suggest, max_evals=200, trials=Trials()))

# -----------------------------------------------

bst_prms_rndm_frst

# ## Xgboost
# -----------------------------------------------

hp_params['model_and_ftrs_space'] = xgb_ftrs

# -----------------------------------------------

bst_prms_xgboost = space_eval(hp_params['model_and_ftrs_space']['features'], fmin(fn=get_model_est_hyper_opt, space=hp_params, algo=tpe.suggest, max_evals=150, trials=Trials()))

# -----------------------------------------------

bst_prms_xgboost

# ## LogisticRegression
# -----------------------------------------------

hp_params['model_and_ftrs_space'] = log_reg_ftrs

# -----------------------------------------------

bst_prms_log_reg = space_eval(hp_params['model_and_ftrs_space']['features'], fmin(fn=get_model_est_hyper_opt, space=hp_params, algo=tpe.suggest, max_evals=150, trials=Trials()))

# ## Checking modelling with other features
# -----------------------------------------------

train_new = [
    feature for feature in train_features if feature not in check_for_use_f
]

# -----------------------------------------------

X = df[train_new]

# -----------------------------------------------

hp_params['model_and_ftrs_space'] = random_forest_ftrs

# -----------------------------------------------

bst_prms_rndm_frst_new = space_eval(hp_params['model_and_ftrs_space']['features'],fmin(fn=get_model_est_hyper_opt, space=hp_params, algo=tpe.suggest, max_evals=200, trials=Trials()))

# -----------------------------------------------

get_model_est(LogisticRegression(), roc_auc_score, df[train_new], df['ill'])

# -----------------------------------------------

get_model_est(
    RandomForestClassifier(**bst_prms_rndm_frst ), roc_auc_score, df[train_features],
    df['ill'])

# -----------------------------------------------

get_model_est_features_enc(RandomForestClassifier(**bst_prms_rndm_frst), roc_auc_score, df[train_new],
                           df['ill'], CatBoostEncoder, cat_features)

# ## Get best model
# -----------------------------------------------

best_model = RandomForestClassifier(**bst_prms_rndm_frst_new)
best_model.fit(df[train_new], df['ill'])

# -----------------------------------------------

roc_auc_score(df.ill, best_model.predict_proba(df[train_new])[:, 1])

# -----------------------------------------------

COST_FN = 5

# -----------------------------------------------

def get_opt_thresh(model, X, y):
    pred_p = model.predict_proba(X)[:, 1]
    threshs = np.arange(0, 1.01, 0.01)
    max_f1 = 0
    opt_thresh = 0
    for thresh in threshs:
        pred_c = pred_p >= thresh
        tn, fp, fn, tp = confusion_matrix(y, pred_c).ravel()
        if tp + fp == 0:
            continue
        recall = tp / (tp + COST_FN * fn)
        precision = tp / (tp + fp)
        f1 = 2 / (1 / recall + 1 / precision)
        if f1 > max_f1:
            max_f1 = f1
            opt_thresh = thresh
    return opt_thresh

# -----------------------------------------------

best_thresh = get_opt_thresh(best_model, df[train_new], df['ill'])

# -----------------------------------------------

best_thresh

# -----------------------------------------------

recall_score(df['ill'], best_model.predict_proba(df[train_new])[:, 1] > best_thresh)

# -----------------------------------------------

precision_score(df['ill'], best_model.predict_proba(df[train_new])[:, 1] > best_thresh)
