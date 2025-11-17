#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter Notebook: 车辆价格预测-baseline.ipynb
# Generated on: 2025-11-17T04:35:10.398Z

# Cell 1 (Markdown)
# # 实验1-xx网汽车交易价格预测

# Cell 2 (Markdown)
# ## 学号：
# ## 姓名：
# ## 班级：

# Cell 3 (Markdown)
# 实验目标：利用提供的汽车交易数据，构建一个能够预测汽车交易价格的模型。通过训练数据训练模型，使用测试数据评估模型的性能，并使用R2作为评价指标。
# 
# 数据列描述：
# 
# 
# | 字段名 | 描述 |
# | :--: | :--: |
# | ID | 交易ID，唯一编码 |
# | name | 汽车交易名称，已脱敏 |
# | regDate | 汽车注册日期，例如20200101，2020年01月01日 |
# | carModel | 车型编码，已脱敏 |
# | brand | 汽车品牌，已脱敏 |
# | BType | 车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7 |
# | FType | 燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6 |
# | gearbox | 变速箱：手动：0，自动：1 |
# | power | 发动机功率：范围 [ 0, 600 ] |
# | km | 汽车已行驶公里，单位万km |
# | notRepaired | 汽车有尚未修复的损坏：是：0，否：1 |
# | regionCode | 地区编码，已脱敏 |
# | seller | 销售方：个体：0，非个体：1 |
# | offerType | 报价类型：提供：0，请求：1 |
# | creatDate | 汽车上线时间，即开始售卖时间 |
# | a系列特征 | 匿名特征，包含a0-9在内10个匿名特征 |
# | price | 二手车交易价格（预测目标y） |
# 
# 使用$R^{2}$[R Squared(r2 score)]进行评估。
# 
# $$
# R^{2}\left( y,\hat{y} \right) = 1 - \frac{\sum_{i = 1}^{m}({\hat{y}}^{\left( i \right)} - \overline{y})^{2}}{\sum_{i = 1}^{m}(y^{\left( i \right)} - \overline{y})^{2}}
# $$
# 
# 提交文件您必须提交一个带有列表ID的csv文件，以及ID的价格。R2值越接近于1，说明模型预测效果越好；R2值越接近于0，说明模型预测效果越差。
# 
# 实验建议步骤：
# 
# 1. 数据准备：
# 
# 
# 	* 下载提供的汽车交易数据集，包含训练集和测试集。
# 	* 探索数据集，了解数据的基本特征，例如汽车的属性、价格等。
# 2. 数据预处理：
# 
# 
# 	* 清洗数据，处理缺失值、异常值和重复值。
# 	* 进行特征工程，提取与汽车价格相关的特征。
# 3. 模型选择：
# 
# 
# 	* 选择合适的回归算法作为预测模型，例如线性回归、决策树回归、随机森林回归或XGBoost回归等。
# 4. 模型训练：
# 
# 
# 	* 使用训练数据集训练所选的回归模型。
# 	* 调整模型的超参数，以优化模型的性能。
# 5. 模型评估：
# 
# 
# 	* 使用测试数据集对训练好的模型进行评估。
# 	* 计算模型的R2分数，评估模型对汽车交易价格的预测能力。
# 6. 结果分析：
# 
# 
# 	* 分析模型的预测结果和R2分数，评估模型的效果。
# 	* 探讨可能影响模型性能的因素，并提出改进建议。
# 7. 提交结果：
# 
# 
# 	* 提交结果参加测评。
# 
# 注意事项：
# 
# * 在实验过程中，确保数据的保密性和隐私性。
# * 遵循实验规范，记录实验过程和结果。
# * 使用合适的工具和库进行数据处理和建模，例如Python的pandas、scikit-learn或XGBoost库等。

# Cell 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Cell 5
## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/testA.csv')

# Cell 6
## 输出数据的大小信息
print('Train data shape:',train.shape)
print('TestA data shape:',test.shape)

# Cell 7
## 通过.head() 简要浏览读取数据的形式
train.head()

# Cell 8 (Markdown)
# ## 学生完成部分
# 特征工程需要自己完成。

# Cell 9
## 通过 .columns 查看列名
train.columns

# Cell 10
numerical_cols = train.select_dtypes(exclude = 'object').columns
print(numerical_cols)

# Cell 11
## 选择特征列
feature_cols = [
    col for col in numerical_cols if col not in [
        'ID', 'name', 'regDate', 'creatDate', 'price', 'model', 'brand',
        'regionCode', 'seller'
    ]
]
feature_cols = [col for col in feature_cols if 'Type' not in col]

## 提前特征列，标签列构造训练样本和测试样本
X_train = train[feature_cols]
y_train =train['price']

X_test = test[feature_cols]

print('X train shape:', X_train.shape)
print('X test shape:', X_test.shape)

# Cell 12
from sklearn.impute import SimpleImputer

# 使用均值填充数值型缺失值
imputer = SimpleImputer(strategy='mean')  # 也可用'median'或'most_frequent'
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)  # 对测试集使用相同的填充值

# Cell 13
import xgboost as xgb
from xgboost import XGBRegressor
# 创建XGBoost模型对象
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
r2_score_train = r2_score(y_train, y_pred_train)
## 因为只有训练集有标签，只能生成训练集的分数
print('R2_score (train): ', r2_score_train)

# Cell 14 (Markdown)
# 后面的代码不建议修改。

# Cell 15
out_df = pd.DataFrame()
out_df['ID'] = test['ID']
out_df['price'] = y_pred_test

# Cell 16 (Markdown)
# 务必把StudentId写成自己的学号，否则没有成绩！

# Cell 17
StudentId = '25451354008'#写自己的学号
subdir=''

# Cell 18
out_df.to_csv(subdir + StudentId + 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
              index=False)

# Cell 19 (Markdown)
# 最后把csv文件上传到ftp://10.132.219.5:955 的相应目录

# Cell 20
