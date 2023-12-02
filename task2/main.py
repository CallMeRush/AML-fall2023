import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_extraction import *
import scipy
from scipy import fft
from scipy import signal
from collections import Counter
import pywt
from biosppy.signals import ecg
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

n_cores = 30
random_state = 71
sampling_rate = 300
data_directory = 'data/'

#class_weights = {0: 3030/5117, 1: 443/5117, 2: 1474/5117, 3: 170/5117}
waveletname = 'db31'
level = 5 #10

X_train_df = pd.read_csv(data_directory + 'X_train.csv', index_col='id')
y_train_df = pd.read_csv(data_directory + 'y_train.csv', index_col='id')
X_test_df = pd.read_csv(data_directory + 'X_test.csv', index_col='id')

def drop_trailing_na(df: pd.DataFrame):
    return [df.loc[i].dropna().to_numpy() for i in range(df.shape[0])]

X_train_full = drop_trailing_na(X_train_df)
y_train_full = y_train_df['y'].to_numpy()
X_test = drop_trailing_na(X_test_df)

def wavelet_transform(signal):
    return pywt.wavedec(signal, waveletname, level=level)

def wavelet_noise_cancellation(signal):
    coeffs = wavelet_transform(signal)
    return pywt.waverec(coeffs, waveletname)

def wavelet_noise_cancellation_bulk(data):
    result = []
    for signal in data:
        result.append(wavelet_noise_cancellation(signal))
    return result

"""X_train_full_filtered = wavelet_noise_cancellation_bulk(X_train_full)
X_test_filtered = wavelet_noise_cancellation_bulk(X_test)"""

def calculate_entropy(list_values):
    value, probabilities = np.unique(list_values, return_counts=True)
    entropy = scipy.stats.entropy(probabilities)
    return [entropy]

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

def get_array_features(arr):
    features = []
    features += calculate_entropy(arr)
    features += calculate_crossings(arr)
    features += calculate_statistics(arr)
    return features

def get_wavelet_features(signal):
    features = []
    list_coeff = wavelet_transform(signal)
    for coeff in list_coeff:
        features += get_array_features(coeff)
    return features

def calculate_consecutive_diff(x):
    return np.ediff1d(x)

def get_values(template, r_peaks, peaks):
    result = []
    for i in range(len(peaks)):
        result.append(template[i][peaks[i] - r_peaks[i] + 60])
    return np.array(result)

def get_ecg_values(signal):
    result = ecg.ecg(signal, sampling_rate=sampling_rate, show=False)
    template = result['templates']

    p_peaks, p_start, p_end = getPPositions(result)

    q_peaks, q_start = getQPositions(result)
    for i in range(len(q_start)):
        if q_start[i] == p_peaks[i]:
            q_start[i] = int(p_end[i] + abs(q_peaks[i] - p_end[i]) / 2)

    r_peaks = result['rpeaks'].tolist()

    s_peaks, s_end = getSPositions(result)
    
    t_peaks, t_start, t_end = getTPositions(result)
    
    beats = fft.fft(template)
    heart_rate = sampling_rate * (60.0 / np.diff(result['rpeaks']))
    heart_rate = np.append(heart_rate, heart_rate[-1]).reshape(-1, 1)

    # They are of length = # heart beats - 1 !!!
    RRinterval = calculate_consecutive_diff(r_peaks)
    PPinterval = calculate_consecutive_diff(p_peaks)
    TPinterval = p_start[1:] - t_end[:-1]

    Pduration = p_end - p_start
    PRsegment = q_start - p_end
    PRinterval = q_start - p_start
    QRScomplex = s_end - q_start
    QTinterval = t_end - q_start
    STsegment = t_start - s_end
    STTsegment = t_end - s_end

    p_values = get_values(template, r_peaks, p_peaks)
    q_values = get_values(template, r_peaks, q_peaks)
    r_values = get_values(template, r_peaks, r_peaks)
    s_values = get_values(template, r_peaks, s_peaks)
    t_values = get_values(template, r_peaks, t_peaks)

    PQ_diff = q_peaks - p_peaks
    PR_diff = r_peaks - p_peaks
    PS_diff = s_peaks - p_peaks
    PT_diff = t_peaks - p_peaks
    QR_diff = r_peaks - q_peaks
    QS_diff = s_peaks - q_peaks
    QT_diff = t_peaks - q_peaks
    RS_diff = s_peaks - r_peaks
    RT_diff = t_peaks - r_peaks
    ST_diff = t_peaks - s_peaks

    return heart_rate, np.real(beats), np.imag(beats), \
        RRinterval, PPinterval, Pduration, PRsegment, PRinterval, QRScomplex, QTinterval, STsegment, STTsegment, TPinterval, \
        p_values, q_values, r_values, s_values, t_values, \
        PQ_diff, PR_diff, PS_diff, PT_diff, QR_diff, QS_diff, QT_diff, RS_diff, RT_diff, ST_diff
    ## Useful values if you want to use time-series-like arrays
    # return p_start, p_peaks, p_end, q_start, q_peaks, r_peaks, s_peaks, s_end, t_start, t_peaks, t_end

def get_features(signal):
    features = []

    ecg_values_list = get_ecg_values(signal)
    for ecg_values in ecg_values_list:
        features += get_array_features(ecg_values)

    features += get_wavelet_features(signal)

    return features

def get_dataset_features(data):
    list_features = []
    for signal in data:
        list_features.append(get_features(signal))
    return list_features

X_train_full_extracted = get_dataset_features(X_train_full)

X_train_extracted, X_val_extracted, y_train, y_val = train_test_split(X_train_full_extracted, y_train_full, test_size=0.2, random_state=random_state)

y_train_01_2 = np.copy(y_train)
y_train_01_2[y_train <= 2] = 0
y_train_01_2[y_train == 3] = 1
X_train_01_oversampled_2, y_train_01_oversampled_2 = SMOTE(random_state=random_state).fit_resample(X_train_extracted, y_train_01_2)
y_val_01_2 = np.copy(y_val)
y_val_01_2[y_val <= 2] = 0
y_val_01_2[y_val == 3] = 1

y_val_01_pred_2 = XGBClassifier(seed=random_state).fit(X_train_01_oversampled_2, y_train_01_oversampled_2.T.ravel()).predict(X_val_extracted)

X_train_012_2 = np.copy(X_train_extracted)[y_train <= 2]
y_train_012_2 = np.copy(y_train)[y_train <= 2]
X_train_012_oversampled_2, y_train_012_oversampled_2 = SMOTE(random_state=random_state).fit_resample(X_train_012_2, y_train_012_2)

X_val_extracted_012_2 = np.copy(X_val_extracted)[y_val_01_pred_2 == 0]
y_val_012_2 = np.copy(y_val)[y_val_01_pred_2 == 0]

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
def find_best_estimator(model, params, X_train, y_train):
    estimator = GridSearchCV(model, params, scoring='f1_micro', cv=5, n_jobs=n_cores, verbose=3)
    estimator.fit(X_train, y_train.T.ravel())
    print(estimator.best_params_)
    return estimator.best_estimator_

from sklearn.svm import SVC
"""svc = SVC(random_state=random_state)
svc_model = Pipeline([ ('scaler', StandardScaler()), ('svc', svc) ])
params = [{
    'svc__C': [10],
    'svc__kernel': ['rbf'],#['rbf', 'poly', 'sigmoid', 'precomputed'], #'linear'
    'svc__degree': [1],#, 2, 3, 4, 5],
    'svc__gamma': ['auto']#, 'scale']
}]
model = Pipeline([ ('scaler', StandardScaler()), ('svc', svc) ])"""
from sklearn.ensemble import ExtraTreesClassifier
"""etr = ExtraTreesClassifier(random_state=random_state)
params = [{
    'etc__n_estimators': [500],
    'etc__min_samples_split': [2],#, 3, 4, 5],
    'etc__min_samples_leaf': [1],#, 2],
    'etc__max_features': [None]
}]
model = Pipeline([ ('scaler', StandardScaler()), ('etc', etr) ])"""
from sklearn.ensemble import GradientBoostingClassifier
"""n_qantiles_full = int(len(X_train_012_oversampled_2)*2/5)
gbc = GradientBoostingClassifier(random_state=random_state)
scaler = QuantileTransformer(output_distribution="normal")
params = [{
    'gbc__n_estimators': [100],
    'gbc__min_samples_split': [2],
    'gbc__min_samples_leaf': [1],
    'gbc__learning_rate': [0.7],
    'gbc__max_depth': [8],
    'gbc__max_features': ['sqrt'],#, None],
    'scaler__n_quantiles': [int(n_qantiles_full/5)]#, int(n_qantiles_full/4)]#, n_qantiles_full]
}]
model = Pipeline([ ('scaler', scaler), ('gbc', gbc) ])"""
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel, ExpSineSquared, RationalQuadratic
k1 = ConstantKernel(constant_value=66.0**2) * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = ConstantKernel(constant_value=2.4**2) * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = ConstantKernel(constant_value=0.66**2) \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = ConstantKernel(constant_value=0.18**2) * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4
gpc = GaussianProcessClassifier(kernel=kernel_gpml, random_state=random_state)
params = [{
    #'gpc__alpha': np.logspace(-2, 4, 5),
    'gpc__kernel__k1__k1__k1__k1__constant_value': np.logspace(-2, 4, 5),
    'gpc__kernel__k1__k1__k1__k2__length_scale': np.logspace(-2, 2, 5),
    'gpc__kernel__k2__k2__noise_level': np.logspace(-2, 1, 5)
}]
model = Pipeline([ ('scaler', StandardScaler()), ('gpc', gpc) ])
"""xgbc = XGBClassifier(seed=random_state)
svc = SVC(C=10, kernel='rbf', degree=1, gamma='auto', random_state=random_state)
svc_model = Pipeline([ ('scaler', StandardScaler()), ('svc', svc) ])
n_quantiles = int(len(X_train_012_oversampled_2)*2/5 / 5)
gbc = GradientBoostingClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, learning_rate=0.7, max_depth=8, max_features=None, random_state=random_state)
scaler = QuantileTransformer(n_quantiles=int(n_quantiles/5), output_distribution="normal")
gbc_model = Pipeline([ ('scaler', scaler), ('gbc', gbc) ])
etc = ExtraTreesClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=random_state)
etc_model = Pipeline([ ('scaler', StandardScaler()), ('etc', etc) ])
from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier(random_state=random_state)
params = [{
    'bc__n_estimators': [10],#, 800],
    'bc__estimator': [svc_model]#[xgbc, svc_model, gbc_model, etc_model]
}]
model = Pipeline([ ('scaler', StandardScaler()), ('bc', bc) ])"""
"""xgbc = XGBClassifier(seed=random_state)
params = [{
    'xgbc__min_child_weight': [1],#, 5, 10],
    'xgbc__gamma': [0.5],#, 1, 1.5, 2, 5],
    'xgbc__subsample': [1.0],#[0.6, 0.8, 1.0],
    'xgbc__colsample_bytree': [0.6],#, 0.8, 1.0],
    'xgbc__max_depth': [3]#, 4, 5]
}]
model = Pipeline([ ('scaler', StandardScaler()), ('xgbc', XGBClassifier(seed=random_state)) ])"""

cls = find_best_estimator(model, params, X_train_012_oversampled_2, y_train_012_oversampled_2)
#cls = XGBClassifier(seed=random_state)
#cls = svc_model
cls.fit(X_train_012_oversampled_2, y_train_012_oversampled_2.T.ravel())

y_train_pred = cls.predict(X_train_012_oversampled_2)
y_val_pred = cls.predict(X_val_extracted_012_2)
train_score = f1_score(y_train_012_oversampled_2, y_train_pred, average='micro')
val_score = f1_score(y_val_012_2, y_val_pred, average='micro')

print(train_score, val_score)
