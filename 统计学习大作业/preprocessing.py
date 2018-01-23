import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import KFold
import pickle

RandomNum = 218

# 处理id和target列
train = pd.read_csv("train.csv")
train_tar = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("test.csv")
test_id = test['id']
del test['id']
# append all data
Alldata = train.append(test)
Alldata.reset_index(inplace=True)

train_rows = train.shape[0]

final_feature = []

# XGboost的参数
params = {
    "eta": 0.1,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "min_child_weight": 55,
    "objective": "reg:linear",
    "booster": "gbtree"
}

for i in ['car', 'ind', 'reg']:
    # fea 为预测目标所需训练数据
    fea = [x for x in list(Alldata) if i not in x]
    train_fea = np.array(Alldata[fea])
    # Alltarget为待预测的变量
    Alltarget = [x for x in list(Alldata) if i in x]
    for target in Alltarget:
        train_tar = Alldata[target]
        # 5折交叉验证,随机分割
        fold = KFold(n_splits=5, random_state=RandomNum, shuffle=True).split(Alldata)
        cv_train = np.zeros(shape=(Alldata.shape[0], 1))
        for i, (train_fd, v) in enumerate(fold):
            X_train, X_v, label_train, label_v = \
                train_fea[train_fd, :], train_fea[v, :], train_tar[train_fd], train_tar[v]
            # 划分测试集和训练集
            dtrain = xgboost.DMatrix(X_train, label_train)
            dvalid = xgboost.DMatrix(X_v, label_v)
            # 设置watchlist
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

            bestModel = xgboost.train(params, dtrain, 500, evals=watchlist, verbose_eval=50, early_stopping_rounds=10)
            cv_train[v, 0] += bestModel.predict(xgboost.DMatrix(X_v), ntree_limit=bestModel.best_ntree_limit)
        # 记录最好的cv结果
        final_feature.append(cv_train)

final_feature = np.hstack(final_feature)
train_fea = final_feature[:train_rows, :]
test_fea = final_feature[train_rows:, :]

# 数据输出，存入pickle中
pickle.dump([train_fea, test_fea], open("preprocessing.pk", 'wb'))
