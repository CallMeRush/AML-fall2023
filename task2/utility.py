n_cores = 1
random_state = 71
data_directory = 'task2/data/'

waveletname = 'db4'

import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import train_test_split, GridSearchCV
import scipy
from collections import Counter
import pywt
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Import data
def import_data(verbose=False):
    if verbose:
        print("Importing data... ", end='')
    X_train_df = pd.read_csv(data_directory + 'X_train.csv', index_col='id')
    y_train_df = pd.read_csv(data_directory + 'y_train.csv', index_col='id')
    X_test_df = pd.read_csv(data_directory + 'X_test.csv', index_col='id')
    if verbose:
        print("data imported.")

    return X_train_df, y_train_df, X_test_df

# Dealing with different lengths
def drop_trailing_na(df: pd.DataFrame):
    return [df.loc[i].dropna().to_numpy() for i in range(df.shape[0])]

def preprocess_data(X_train_df, y_train_df, X_test_df, verbose=False):
    if verbose:
        print("Preprocessing data... ", end='')
    X_train_full = drop_trailing_na(X_train_df)
    y_train_full = y_train_df['y'].to_numpy()
    X_test = drop_trailing_na(X_test_df)
    if verbose:
        print("data preprocessed.")

    return X_train_full, y_train_full, X_test

# Parallel import and preprocess
def parallel_import_preprocess_X(filename):
    return drop_trailing_na(pd.read_csv(data_directory + filename, index_col='id'))

def parallel_import_preprocess():
    p = Pool(min(2, n_cores))
    X_train_full, X_test = p.map(parallel_import_preprocess_X, ['X_train.csv', 'X_test.csv'])
    p.close()
    p.join()

    y_train_full = pd.read_csv(data_directory + 'y_train.csv', index_col='id')['y'].to_numpy()

    return X_train_full, y_train_full, X_test


# Split training and validation
def split_train_val(X_train_full, y_train_full):
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=random_state)
    return X_train, X_val, y_train, y_val


# Features to extract
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]
 
def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

# Noise handling
def wavelet_transform(signal, waveletname=waveletname):
    return pywt.wavedec(signal, waveletname, level=7)

# Feature extraction
def get_dataset_features(data, waveletname=waveletname):
    list_features = []
    for signal in data:
        list_coeff = wavelet_transform(signal, waveletname=waveletname)
        features = []
        for coeff in list_coeff:
            features += get_features(coeff)
        list_features.append(features)
    return list_features


# Classifier definition
def get_classifier():
    cls = GradientBoostingClassifier(n_estimators=100, verbose=1, random_state=random_state)
    cls = XGBClassifier(seed=random_state)
    return cls

def find_best_classifier(model, params, X_train, y_train):
    classifier = GridSearchCV(model, params, scoring='f1_micro', cv=5, n_jobs=n_cores, verbose=3)
    classifier.fit(X_train, y_train.T.ravel())
    print(classifier.best_params_)
    return classifier.best_estimator_


# Training with validation
def print_f1_score(y_train, y_train_pred, y_val, y_val_pred):
    train_score = f1_score(y_train, y_train_pred, average='micro')
    val_score = f1_score(y_val, y_val_pred, average='micro')
    print("F1 score: ", train_score, val_score)
    return val_score

def train_with_val(cls, X_train_extracted, y_train, X_val_extracted, y_val):
    cls.fit(X_train_extracted, y_train.T.ravel())

    y_train_pred = cls.predict(X_train_extracted)
    y_val_pred = cls.predict(X_val_extracted)

    val_score = print_f1_score(y_train, y_train_pred, y_val, y_val_pred)
    return val_score


# Final training
def final_training(cls, X_train_full_extracted, y_train_full, X_test_extracted):
    y_test_pred = cls.fit(X_train_full_extracted, y_train_full.T.ravel()).predict(X_test_extracted)
    return y_test_pred


# Write result
def write_result(y_test_pred):
    table = pd.DataFrame({'id': np.arange(0, y_test_pred.shape[0]), 'y': y_test_pred.flatten()})
    table.to_csv(data_directory + 'y_test_pred.csv', index=False)
