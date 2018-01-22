#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:13:30 2018

@author: leonard_tia

Title:Lasso and Elastic Net for Sparse Signals
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# #############################################################################
# 生成一些稀疏的数据来使用
np.random.seed(42)

n_samples, n_features = 50, 200
#生成一个随机的50行200列的数组
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
#混洗index
np.random.shuffle(inds)
coef[inds[10:]] = 0  # 创建稀疏系数，把coef里的随机的190个系数设为0
y = np.dot(X, coef)#X样本与稀疏系数点乘

# 加入噪音
y += 0.01 * np.random.normal(size=n_samples)

# 在训练集和测试集中分割数据
n_samples = X.shape[0]
#前25个样本为训练集
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
#后25个样本为测试集
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

# #############################################################################
# Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
#LASSO拟合的y
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
#利用可决系数评定模型的分数，满分1.0
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

# #############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()