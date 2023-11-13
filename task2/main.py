import pandas as pd

def import_data():
    X_train_df = pd.read_csv('task2/data/X_train.csv', index_col='id')
    y_train_df = pd.read_csv('task2/data/y_train.csv', index_col='id')
    X_test_df = pd.read_csv('task2/data/X_test.csv', index_col='id')

    X_train = X_train_df.loc[0].dropna().to_numpy(dtype='float32')
    y_train = y_train_df.loc[0]
    X_test = X_test_df.loc[0].dropna().to_numpy(dtype='float32')

    return X_train, y_train, X_test


X_train, y_train, X_test = import_data()

print(X_train.shape)
