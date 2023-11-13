n_cores = 16
random_state = 71

# Data handling
import numpy as np
import pandas as pd

def import_data():
    X_train_df = pd.read_csv('task1/data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('task1/data/y_train.csv', skiprows=1, header=None)
    X_test_df = pd.read_csv('task1/data/X_test.csv', skiprows=1, header=None)

    X_train = X_train_df.values[:, 1:]
    y_train = y_train_df.values[:, 1:]
    X_test = X_test_df.values[:, 1:]

    print(y_train)

    return X_train, y_train, X_test


# Filling missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

def fill_nan(X):
    #return SimpleImputer(strategy="median").fit_transform(X)
    #return KNNImputer(n_neighbors=5).fit_transform(X)
    return IterativeImputer(random_state=random_state, verbose=2, estimator=get_svr(X, None)).fit_transform(X)


# Outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def outlier_removal(X_train, y_train):
    # (same number of outliers, but 10% are different)
    #X_train_w_outliers = fill_nan(X_train)
    X_train_w_outliers = X_train

    #outliers = IsolationForest(max_samples=100, random_state=random_state, contamination='auto').fit_predict(X_train_w_outliers)
    outliers = LocalOutlierFactor(n_neighbors=20, contamination='auto').fit_predict(X_train_w_outliers)

    X_train = X_train[outliers == 1]
    y_train = y_train[outliers == 1]

    return X_train, y_train


# Preprocessing
def preprocessing():
    X_train, y_train, X_test = import_data()

    X_train = fill_nan(X_train)
    X_test = fill_nan(X_test)

    X_train, y_train = outlier_removal(X_train, y_train)

    return X_train, y_train, X_test


# Feature selection
from sklearn.feature_selection import SelectKBest, f_regression

def select_features(X_train, y_train, X_test, k=189):
    """selected_features = np.loadtxt('task1/selected_features_2.txt', dtype=bool)
    X_train = X_train[:, selected_features]
    X_test = X_test[:, selected_features]"""

    selection = SelectKBest(f_regression, k=k).fit(X_train, y_train.T.ravel())
    X_train = selection.transform(X_train)
    X_test = selection.transform(X_test)

    return X_train, X_test


# Print r2 score
from sklearn.metrics import r2_score

def print_r2_score(regressor, X_train, X_val, y_train, y_val):
    y_train_pred = regressor.predict(X_train)
    y_val_pred = regressor.predict(X_val)

    train_score = r2_score(y_train, y_train_pred)
    val_score = r2_score(y_val, y_val_pred)

    print(train_score, val_score)


# Training and validation
from sklearn.model_selection import train_test_split, GridSearchCV

def find_best_estimator(model, params, X_train, y_train):
    estimator = GridSearchCV(model, params, scoring='r2', cv=5, n_jobs=n_cores, verbose=3)
    estimator.fit(X_train, y_train.T.ravel())
    print(estimator.best_params_)
    return estimator.best_estimator_

def train_with_validation(model, params, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    regressor = find_best_estimator(model, params, X_train, y_train)
    print_r2_score(regressor, X_train, X_val, y_train, y_val)
    return regressor

def train_best(regressor, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    regressor.fit(X_train, y_train.T.ravel())
    print_r2_score(regressor, X_train, X_val, y_train, y_val)
    return regressor


# Get the different models, with best parameters
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR
def get_svr(X_train, y_train):
    svr = SVR()

    # Params search
    params = [{
        'svr__C': [75],#np.logspace(0, 3, 4),
        'svr__epsilon': [0.003],#np.logspace(-6, 0, 7),
        'svr__kernel': ['rbf', 'poly'],#['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'svr__degree': [1, 2, 3, 4, 5],
        'svr__gamma': ['auto']
    }]
    # Best params
    best_params = {
        'C': 75,
        'epsilon': 0.003,
        'kernel': 'rbf',
        'gamma': 'auto'
    }
    svr.set_params(**best_params)

    model = Pipeline([ ('scaler', StandardScaler()), ('svr', svr) ])
    #return train_with_validation(model, params, X_train, y_train)
    #return train_best(model, X_train, y_train)
    return model

from sklearn.ensemble import GradientBoostingRegressor
def get_gb(X_train, y_train):
    cv_val = 5
    n_qantiles_full = int(X_train.shape[0]*2/cv_val)
    gb = GradientBoostingRegressor(random_state=random_state)
    scaler = QuantileTransformer(output_distribution="normal")

    # Params search
    params = [{
        'gb__n_estimators': [10000],
        'gb__min_samples_split': [2],
        'gb__min_samples_leaf': [1],
        'gb__learning_rate': [0.1],
        'gb__max_depth': [3],
        'gb__max_features': ['sqrt'],
        'scaler__n_quantiles': [int(n_qantiles_full/2), n_qantiles_full]
    }]
    # Best params
    best_params_scaler = { 'n_quantiles': int(n_qantiles_full/2) }
    scaler.set_params(**best_params_scaler)
    best_params_gb = {
        'n_estimators': 3000,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'learning_rate': 0.01,
        'max_depth': 3,
        'max_features': 'sqrt',
    }
    gb.set_params(**best_params_gb)

    model = Pipeline([ ('scaler', scaler), ('gb', gb) ])
    #return train_with_validation(model, params, X_train, y_train)
    #return train_best(model, X_train, y_train)
    return model

from sklearn.ensemble import ExtraTreesRegressor
def get_etr(X_train, y_train):
    etr = ExtraTreesRegressor(random_state=random_state)

    # Params search
    params = [{
        'etr__n_estimators': [438, 440, 442, 444],
        'etr__min_samples_split': [2],
        'etr__min_samples_leaf': [1],
        'etr__max_features': ['sqrt']
    }]
    # Best params
    best_params = {
        'n_estimators': 440,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    }
    etr.set_params(**best_params)

    model = Pipeline([ ('scaler', StandardScaler()), ('etr', etr) ])
    #return train_with_validation(model, params, X_train, y_train)
    #return train_best(model, X_train, y_train)
    return model

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel, ExpSineSquared, RationalQuadratic
def get_gpr(X_train, y_train):
    # Kernel with parameters given in GPML book
    k1 = ConstantKernel(constant_value=66.0**2) * RBF(length_scale=67.0)  # long term smooth rising trend
    k2 = ConstantKernel(constant_value=2.4**2) * RBF(length_scale=90.0) \
        * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
    # medium term irregularity
    k3 = ConstantKernel(constant_value=0.66**2) \
        * RationalQuadratic(length_scale=1.2, alpha=0.78)
    k4 = ConstantKernel(constant_value=0.18**2) * RBF(length_scale=0.134) \
        + WhiteKernel(noise_level=0.19**2)  # noise terms
    kernel_gpml = k1 + k2 + k3 + k4
    gp = GaussianProcessRegressor(kernel=kernel_gpml, random_state=random_state)

    # Params search
    params = [{
        'gpr__alpha': [0.3, 0.4],#np.logspace(-2, 4, 5),
        'gpr__kernel__k1__k1__k1__k1__constant_value': [12000],#[10000, 30000],
        'gpr__kernel__k1__k1__k1__k2__length_scale': [2.6],#[0.5, 1, 2],
        'gpr__kernel__k2__k2__noise_level': [20]#np.logspace(-2, 1, 5)
    }]
    # Best params
    best_params_kernel = {
        'k1__k1__k1__k1__constant_value': 12000, 
        'k1__k1__k1__k2__length_scale': 2.6, 
        'k2__k2__noise_level': 20
    }
    kernel_gpml.set_params(**best_params_kernel)
    best_params_gpr = { 'alpha': 0.31622776601683794 }
    gp.set_params(**best_params_gpr)

    model = Pipeline([ ('scaler', StandardScaler()), ('gpr', gp) ])
    #return train_with_validation(model, params, X_train, y_train)
    #return train_best(model, X_train, y_train)
    return model

from sklearn.ensemble import BaggingRegressor
def get_br(X_train, y_train, estimators, estimator=None):
    br = BaggingRegressor(random_state=random_state)

    # Params search
    params = [{
        'br__n_estimators': [300, 800],
        'br__estimator': estimators
    }]
    # Best params
    best_params = {
        'n_estimators': 500,
        'estimator': estimator
    }
    br.set_params(**best_params)

    model = Pipeline([ ('scaler', StandardScaler()), ('br', br) ])
    #return train_with_validation(model, params, X_train, y_train)
    #return train_best(model, X_train, y_train)
    return model

from sklearn.ensemble import AdaBoostRegressor
def get_abr(X_train, y_train, estimators, estimator=None):
    abr = AdaBoostRegressor(random_state=random_state)

    # Params search
    params = [{
        'abr__n_estimators': [100, 500],
        'abr__estimator': estimators
    }]
    # Best params
    best_params = {
        'n_estimators': 100,
        'estimator': estimator
    }
    abr.set_params(**best_params)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('abr', abr)
    ])
    #return train_with_validation(model, params, X_train, y_train)
    #return train_best(model, X_train, y_train)
    return model


# Final training and validation
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV

def get_final_regressor(estimators):
    final_pipeline = Pipeline([ ('model', LassoCV()) ])
    regressor = StackingRegressor(estimators, final_pipeline, n_jobs=n_cores)
    return regressor

def final_train_with_validation(regressor, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    regressor.fit(X_train, y_train.T.ravel())
    print_r2_score(regressor, X_train, X_val, y_train, y_val)
    

# Writing predictions
def write_result(regressor, X_train, y_train, X_test):
    y_test_pred = regressor.fit(X_train, y_train.T.ravel()).predict(X_test)
    table = pd.DataFrame({'id': np.arange(0, y_test_pred.shape[0]), 'y': y_test_pred.flatten()})
    table.to_csv('task1/data/y_test_pred_2.csv', index=False)
