import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import lightgbm
from scipy import sparse

RandomNumber = 200

train = pd.read_csv('train.csv')
train_target = train['target']
train_id = train['id']
test = pd.read_csv('test.csv')
test_id = test['id']

X = train.drop(['id', 'target'], axis=1)
Allfeature = X.columns.tolist()

# 找出所有定类变量
cat = [x for x in Allfeature if ('cat' in x and 'count' not in x)]

# 找出所有数值变量
num = [x for x in Allfeature if ('cat' not in x and 'calc' not in x)]

# 统计缺失值
train['miss'] = (train == -1).sum(axis=1).astype(float)
test['miss'] = (test == -1).sum(axis=1).astype(float)
num.append('miss')

# encoder
for item in cat:
    target = LabelEncoder()
    target.fit(train[item])
    train[item] = target.transform(train[item])
    test[item] = target.transform(test[item])

# 做onehot变换
onehot = OneHotEncoder()
onehot.fit(train[cat])
X_cat = onehot.transform(train[cat])
test_cat = onehot.transform(test[cat])

# 找出所有individual变量
ind = [x for x in Allfeature if 'ind' in x]
count = 0
for item in ind:
    if count == 0:
        train['new_ind'] = train[item].astype(str) + '_'
        test['new_ind'] = test[item].astype(str) + '_'
        count += 1
    else:
        train['new_ind'] += train[item].astype(str) + '_'
        test['new_ind'] += test[item].astype(str) + '_'

cat_cnt = []
for item in cat + ['new_ind']:
    dict = pd.concat([train[item], test[item]]).value_counts().to_dict()
    train['%s_count' % item] = train[item].apply(lambda x: dict.get(x, 0))
    test['%s_count' % item] = test[item].apply(lambda x: dict.get(x, 0))
    cat_cnt.append('%s_count' % item)


def My_Gini(y_true, y_prediction):
    n = y_true.shape[0]

    # 分别按照真实值和预测值排序
    order = np.array([y_true, y_prediction]).transpose()
    prediction = order[order[:, 1].argsort()][::-1, 0]
    true = order[order[:, 0].argsort()][::-1, 0]

    # 计算基尼系数
    Gini_true = np.sum(np.linspace(1 / n, 1, n) - np.cumsum(true) * 1. / np.sum(true))
    Gini_prediction = np.sum(np.linspace(1 / n, 1, n) - np.cumsum(true) * 1. / np.sum(true))

    # 归一化
    return (Gini_prediction * 1. / Gini_true)


def calerr(predictions, dtrain):
    l = dtrain.get_target()
    return 'My_Gini', My_Gini(l, predictions), True


train_all = [train[num + cat_cnt].values, X_cat, ]
test_all = [test[num + cat_cnt].values, test_cat, ]

X = sparse.hstack(train_all).tocsr()
X_test = sparse.hstack(test_all).tocsr()
# set lightgbm parameters
params = {
    "learning_rate": 0.1,
    "num_leaves": 15,
    "feature_fraction": 0.6,
    "max_bin": 256,
    "is_unbalance": False,
    "max_drop": 50,
    "verbosity": 0,
    "drop_rate": 0.1,
    "min_child_samples": 10,
    "min_child_weight": 150,
    "min_split_gain": 0,
    "subsample": 0.9,
    "objective": "binary",
    "boosting_type": "gbdt",
}

gini = []
final_train = np.zeros(len(train_target))
final_prediction = np.zeros(len(test_id))
for seed in range(20):
    params['seed'] = seed
    cv_train = np.zeros(len(train_target))
    cv_prediction = np.zeros(len(test_id))
    best = []
    f_gini = []


    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RandomNumber).split(X, train_target)

    for i, (train_fold, v) in enumerate(kfold):
        X_train, X_v, target_train, target_v = X[train_fold, :], X[v, :], train_target[train_fold], train_target[v]
        dtrain = lightgbm.Dataset(X_train, target_train)
        dvalid = lightgbm.Dataset(X_v, target_v, reference=dtrain)
        best_model = lightgbm.train(params, dtrain, 10000, valid_sets=dvalid, feval=calerr, verbose_eval=100,
                                    early_stopping_rounds=100)
        best.append(best_model.best_iteration)

        cv_train[v] += best_model.predictionict(X_v)
        cv_prediction += best_model.predictionict(X_test, num_iteration=best_model.best_iteration)

        gini = My_Gini(target_v, cv_train[v])
        f_gini.append(gini)

    cv_prediction /= 5
    final_prediction += cv_prediction
    final_train += cv_train
    gini.append(My_Gini(train_target, cv_train))

what = pd.DataFrame({'id': test_id, 'target': final_prediction / 20})
what.to_csv('answer.csv', index=False)


