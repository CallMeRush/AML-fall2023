from utility import *

wavelets = pywt.wavelist(kind='discrete')
#wavelets = ['db13', 'db16', 'db19', 'db26', 'db27', 'sym6', 'sym9']

X_train_full, y_train_full, X_test = parallel_import_preprocess()
X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_train_full)

def func(w):
    X_train_extracted = get_dataset_features(X_train, waveletname=w)
    X_val_extracted = get_dataset_features(X_val, waveletname=w)

    cls = get_classifier()

    val_score = train_with_val(cls, X_train_extracted, y_train, X_val_extracted, y_val)
    return w, val_score

from multiprocessing import Pool
p = Pool(n_cores)
results = p.map(func, wavelets)
p.close()
p.join()

scores = [x[0] for x in results]
wavelets = [x[1] for x in results]

for i in range(len(wavelets)):
    print(f"{i}) {wavelets[i]} \t\t {scores[i]}")
print(f"Best: {max(scores)} achieved by {wavelets[scores.index(max(scores))]}")


"""scores = []
for i in range(len(wavelets)):
    #print("%d) %s" % (i, wavelets[i]))

    X_train_extracted = get_dataset_features(X_train, waveletname=wavelets[i])
    X_val_extracted = get_dataset_features(X_val, waveletname=wavelets[i])
    #X_train_full_extracted = get_dataset_features(X_train_full, waveletname=wavelets[i])

    cls = get_classifier()
    #cls = find_best_classifier(model, params, X_train_full_extracted, y_train_full)

    val_score = train_with_val(cls, X_train_extracted, y_train, X_val_extracted, y_val)
    scores.append(val_score)

for i in range(len(wavelets)):
    print("%d) %s \t\t %f" % (i, wavelets[i], scores[i]))
print("Best: %f achieved by %s" % (max(scores), wavelets[scores.index(max(scores))]))"""

"""X_train_full_extracted = get_dataset_features(X_train_full)
X_test_extracted = get_dataset_features(X_test)

y_test_pred = final_training(cls, X_train_full_extracted, y_train_full, X_test_extracted)

write_result(y_test_pred)"""
