from utility import *

# LocalOutlierFactor
# k = 189 : 0.9999999991261346 0.563446169382984
#           0.9490584976811819 0.4831137835513042
#           1.0                0.46137830360951315
#           0.9919537029625023 0.5722817324100093

# IsolationForest (or 176)
# k = 178 : 0.9999999858726142 0.557869709786528
#           0.9308627989337217 0.5281897795070347
#           1.0                0.46894067163796727
#           0.9968994872653001 0.5724993566954755

X_train, y_train, X_test = preprocessing()

X_train, X_test = select_features(X_train, y_train, X_test, k=189)

svr_model = get_svr(X_train, y_train)
gb_model = get_gb(X_train, y_train)
etr_model = get_etr(X_train, y_train)
gpr_model = get_gpr(X_train, y_train)

estimators = [svr_model] #[svr_model, gb_model, etr_model, gpr_model]
br_model = get_br(X_train, y_train, estimators, svr_model)
abr_model = get_abr(X_train, y_train, estimators, svr_model)

estimators = [
    ('svr', svr_model),
    ('gb', gb_model),
    ('etr', etr_model),
    ('gpr', gpr_model),
    ('br', br_model),
    ('abr', abr_model)
]

regressor = get_final_regressor(estimators)

#final_train_with_validation(regressor, X_train, y_train)

write_result(regressor, X_train, y_train, X_test)
