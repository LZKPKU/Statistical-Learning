import numpy as np
import pandas as pd
import os
import seaborn as sns


trainFile = "E:/My University/My Course/2017-2018第一学期/统计学习/大作业/train.csv"
testFile = "E:/My University/My Course/2017-2018第一学期/统计学习/大作业/test.csv"

os.chdir(os.path.dirname(trainFile))
train = pd.read_csv(os.path.basename(trainFile))
test = pd.read_csv(os.path.basename(testFile))

train_y = train.target
del train["target"],train["id"],test["id"]

data = train.append(test)
# histplot
for x in data.columns:
    fig = sns.distplot(data[x],kde=False)
    sns.plt.savefig(x)
    sns.plt.close()

# violionplot
d = sns.violinplot(x=data["ps_reg_03"])
sns.plt.savefig("violin")

# jointplot
g = sns.jointplot(data.ps_calc_10,data.ps_calc_11,
                    size=5, ratio=3, color="g")
sns.plt.savefig("jointplot")

# correlation coefficient matrix
cor = np.corrcoef(data,rowvar=0)

cor = pd.DataFrame(cor)
# We have the coefficient matrix
cor.to_csv("cor.csv")


