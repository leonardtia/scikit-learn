#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:21:30 2018
titile：Ordinary Least Squares
@author: leonard_tia
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 载入糖尿病数据集；
diabetes = datasets.load_diabetes()

# 只使用一个特性；data里的第3列作为特性
diabetes_X =diabetes.data[:, np.newaxis,2]

# 将数据分成培训/测试集
diabetes_X_train =diabetes_X[:-20]
diabetes_X_test =diabetes_X[-20:]

# 将数据分成培训/测试集
diabetes_y_train =diabetes.target[:-20]
diabetes_y_test =diabetes.target[-20:]

# 创建线性回归对象
regr = linear_model.LinearRegression()

# 利用训练集对模型进行训练
regr.fit(diabetes_X_train,diabetes_y_train)

# 使用测试集进行预测
diabetes_y_pred =regr.predict(diabetes_X_test)

# 系数
print('Coefficients: \n',regr.coef_)
# 均方误差
print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test,diabetes_y_pred))
# 解释方差得分：1是完全预测。
print('Variance score: %.2f' % r2_score(diabetes_y_test,diabetes_y_pred))

# 绘图输出
plt.scatter(diabetes_X_test,diabetes_y_test,color='black')
plt.plot(diabetes_X_test,diabetes_y_pred,color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

reg.coef_

