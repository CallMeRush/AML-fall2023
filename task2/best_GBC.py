from utility import *

wavelet = 'db13' #'sym9'

if __name__ == '__main__':
    X_train_full, y_train_full, X_test = parallel_import_preprocess()
    print(type(X_train_full))
    X_train_full_extracted = get_dataset_features(X_train_full, waveletname=wavelet)

    from sklearn.preprocessing import QuantileTransformer
    cv_val = 5
    n_qantiles_full = int(X_train_full_extracted.shape[0]*2/cv_val)
    gb = GradientBoostingClassifier(random_state=random_state, verbose=1)
    scaler = QuantileTransformer(output_distribution="normal")

    from sklearn.pipeline import Pipeline
    params = [{
        'gb__n_estimators': [3000], # 3000 better than 1000
        'gb__min_samples_split': [3], # 3 better than 2
        'gb__min_samples_leaf': [1], # 1 better than 2
        'gb__learning_rate': [0.1], # 0.1 better than 0.01, 1.0
        'gb__max_depth': [4, 5], # 4 better than 3
        'gb__max_features': ['sqrt'], #, 'log2'],
        'scaler__n_quantiles': [int(n_qantiles_full/2)] #, n_qantiles_full]
    }]
    model = Pipeline([ 
        ('scaler', scaler), 
        ('gb', gb) 
    ])

    cls = find_best_classifier(model, params, X_train_full_extracted, y_train_full)

    X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_train_full)
    X_train_extracted = get_dataset_features(X_train, waveletname=wavelet)
    X_val_extracted = get_dataset_features(X_val, waveletname=wavelet)
    train_with_val(cls, X_train_extracted, y_train, X_val_extracted, y_val)


    """X_train_full_extracted = get_dataset_features(X_train_full)
    X_test_extracted = get_dataset_features(X_test)

    y_test_pred = final_training(cls, X_train_full_extracted, y_train_full, X_test_extracted)

    write_result(y_test_pred)"""
