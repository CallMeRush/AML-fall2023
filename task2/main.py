from utility import *

X_train_df, y_train_df, X_test_df = import_data()

X_train_full, y_train_full, X_test = preprocess_data(X_train_df, y_train_df, X_test_df)

X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_train_full)

X_train_extracted = get_dataset_features(X_train)
X_val_extracted = get_dataset_features(X_val)

cls = get_classifier()

train_with_val(cls, X_train_extracted, y_train, X_val_extracted, y_val)

X_train_full_extracted = get_dataset_features(X_train_full)
X_test_extracted = get_dataset_features(X_test)

y_test_pred = final_training(cls, X_train_full_extracted, y_train_full, X_test_extracted)

write_result(y_test_pred)
