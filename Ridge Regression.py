#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:28:12 2018

@author: leonard_tia
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X 是 10x10 希尔伯特矩阵
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# #############################################################################
# 计算路径

n_alphas = 200
#返回以n_alphas作为num,以对数作为刻度的数组
alphas = np.logspace(-10, -2, n_alphas)￼

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# #############################################################################
# 显示结果

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')#对数缩放X轴
ax.set_xlim(ax.get_xlim()[::-1])  # 设定X轴的起始大小和结束大小
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
reg.coef_
#用交叉验证法进行正则化参数选择
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.alpha_