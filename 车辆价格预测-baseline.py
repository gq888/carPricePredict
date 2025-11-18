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

# 导入LightGBM
import lightgbm as lgb
from lightgbm import LGBMRegressor

# 导入CatBoost
from catboost import CatBoostRegressor

# 导入RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# 导入必要的库
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import mutual_info_regression, SelectKBest, RFE
from sklearn.linear_model import LinearRegression

# 简单的数据增强：随机采样增强
from sklearn.utils import resample

# 使用RandomizedSearchCV代替GridSearchCV进行更全面的超参数搜索
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# 导入所需的评估指标
from sklearn.metrics import mean_absolute_error

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

# 时间特征提取
print('\n开始时间特征提取...')

# 处理日期转换的辅助函数
def process_date(date_str, format='%Y%m%d'):
    try:
        return pd.to_datetime(date_str, format=format)
    except (ValueError, TypeError):
        # 处理特殊情况：月份为00的情况（如20110010）
        if len(str(date_str)) == 8:
            year = str(date_str)[:4]
            month = str(date_str)[4:6]
            day = str(date_str)[6:8]
            
            # 如果月份为00，设置为01
            if month == '00':
                month = '01'
                corrected_date = f"{year}{month}{day}"
                try:
                    return pd.to_datetime(corrected_date, format=format)
                except:
                    return pd.NaT
            
            # 处理20220229（2022年不是闰年）的情况
            if date_str == '20220229':
                return pd.to_datetime('20220228', format=format)
        return pd.NaT

# 转换日期列
train['regDate_str'] = train['regDate'].astype(str)
train['regDate'] = train['regDate_str'].apply(process_date)

# 对测试集进行相同处理
test['regDate_str'] = test['regDate'].astype(str)
test['regDate'] = test['regDate_str'].apply(process_date)

train['creatDate_str'] = train['creatDate'].astype(str)
train['creatDate'] = train['creatDate_str'].apply(process_date)

test['creatDate_str'] = test['creatDate'].astype(str)
test['creatDate'] = test['creatDate_str'].apply(process_date)

# 查看转换结果
print(f"regDate转换成功: {train['regDate'].notna().sum()}/{len(train)}")
print(f"creatDate转换成功: {train['creatDate'].notna().sum()}/{len(train)}")

# 清理临时字符串列
train = train.drop(['regDate_str', 'creatDate_str'], axis=1)
test = test.drop(['regDate_str', 'creatDate_str'], axis=1)

# 从regDate提取特征：年份、月份、季度
train['reg_year'] = train['regDate'].dt.year
train['reg_month'] = train['regDate'].dt.month
train['reg_quarter'] = train['regDate'].dt.quarter

# 从creatDate提取特征：年份、月份、季度、星期几
train['creat_year'] = train['creatDate'].dt.year
train['creat_month'] = train['creatDate'].dt.month
train['creat_quarter'] = train['creatDate'].dt.quarter
train['creat_dayofweek'] = train['creatDate'].dt.dayofweek

# 计算车辆使用年限（天）
train['car_age_days'] = (train['creatDate'] - train['regDate']).dt.days

# 对测试集进行相同的特征提取
test['reg_year'] = test['regDate'].dt.year
test['reg_month'] = test['regDate'].dt.month
test['reg_quarter'] = test['regDate'].dt.quarter

test['creat_year'] = test['creatDate'].dt.year
test['creat_month'] = test['creatDate'].dt.month
test['creat_quarter'] = test['creatDate'].dt.quarter
test['creat_dayofweek'] = test['creatDate'].dt.dayofweek

test['car_age_days'] = (test['creatDate'] - test['regDate']).dt.days

# 查看新添加的时间特征
time_features = ['reg_year', 'reg_month', 'reg_quarter', 'creat_year', 'creat_month', 'creat_quarter', 'creat_dayofweek', 'car_age_days']
print('添加的时间特征:', time_features)
print('时间特征示例:')
print(train[time_features].head())

# Cell 10
# 查看原始数据中的品牌、型号和地域列信息
print('原始数据中的品牌、型号和地域列信息：')
print('品牌(brand) - 唯一值数量:', train['brand'].nunique())
print('型号(carModel) - 唯一值数量:', train['carModel'].nunique())
print('地域(regionCode) - 唯一值数量:', train['regionCode'].nunique())

# 查看这些列的一些基本统计信息
print('\n品牌、型号和地域的基本统计信息：')
print('品牌分布：', train['brand'].value_counts().head(10))
print('型号分布：', train['carModel'].value_counts().head(10))
print('地域分布：', train['regionCode'].value_counts().head(10))

numerical_cols = train.select_dtypes(exclude = 'object').columns
print('\n数值列：', numerical_cols)

# Cell 11
## 选择特征列 - 包含新的时间特征和Type特征
feature_cols = [
    col for col in numerical_cols if col not in [
        'ID', 'name', 'regDate', 'creatDate', 'price', 'carModel', 'brand',
        'regionCode', 'seller'
    ]
]
# 重新纳入Type特征
feature_cols.extend(['BType', 'FType'])

## 异常值处理
# power特征存在严重异常值，进行处理
print('\n异常值处理前:')
print(f'power最大值: {train["power"].max()}')
print(f'power大于600的样本数: {len(train[train["power"] > 600])}')

# 使用99.9%分位数截断power异常值
power_upper = train['power'].quantile(0.999)
train.loc[train['power'] > power_upper, 'power'] = power_upper
test.loc[test['power'] > power_upper, 'power'] = power_upper

print('\n异常值处理后:')
print(f'power最大值: {train["power"].max()}')
print(f'power大于600的样本数: {len(train[train["power"] > 600])}')
print(f'使用的截断值: {power_upper}')

# Cell 12
## Cell 13
print('\n=== 特征组合与特征选择 ===')

# 1. 创建特征组合
print('\n1. 创建特征组合')

# 现有特征组合
train['power_km_ratio'] = train['power'] / (train['km'] + 1)  # +1避免除零
test['power_km_ratio'] = test['power'] / (test['km'] + 1)

train['power_age_ratio'] = train['power'] / (train['car_age_days'] + 1)
test['power_age_ratio'] = test['power'] / (test['car_age_days'] + 1)

train['km_age_ratio'] = train['km'] / (train['car_age_days'] + 1)
test['km_age_ratio'] = test['km'] / (test['car_age_days'] + 1)

# 基于品牌、型号和地域的组合特征
print('\n创建基于品牌、型号和地域的组合特征...')

# 1. 品牌-型号组合
print('创建品牌-型号组合特征...')
train['brand_model'] = train['brand'].astype(str) + '_' + train['carModel'].astype(str)
test['brand_model'] = test['brand'].astype(str) + '_' + test['carModel'].astype(str)

# 2. 品牌-地域组合
print('创建品牌-地域组合特征...')
train['brand_region'] = train['brand'].astype(str) + '_' + train['regionCode'].astype(str)
test['brand_region'] = test['brand'].astype(str) + '_' + test['regionCode'].astype(str)

# 3. 型号-地域组合
print('创建型号-地域组合特征...')
train['model_region'] = train['carModel'].astype(str) + '_' + train['regionCode'].astype(str)
test['model_region'] = test['carModel'].astype(str) + '_' + test['regionCode'].astype(str)

# 4. 品牌平均功率特征
print('创建品牌平均功率特征...')
brand_avg_power = train.groupby('brand')['power'].mean().to_dict()
train['brand_avg_power'] = train['brand'].map(brand_avg_power)
test['brand_avg_power'] = test['brand'].map(brand_avg_power)

# 5. 地域平均车龄特征
print('创建地域平均车龄特征...')
region_avg_age = train.groupby('regionCode')['car_age_days'].mean().to_dict()
train['region_avg_age'] = train['regionCode'].map(region_avg_age)
test['region_avg_age'] = test['regionCode'].map(region_avg_age)

# 6. 品牌-型号平均价格特征（用于参考，不直接作为特征）
print('创建品牌-型号平均价格特征...')
brand_model_avg_price = train.groupby(['brand', 'carModel'])['price'].mean().to_dict()
train['brand_model_avg_price'] = train.set_index(['brand', 'carModel']).index.map(brand_model_avg_price)
test['brand_model_avg_price'] = test.set_index(['brand', 'carModel']).index.map(brand_model_avg_price)

# 将新特征转换为数值类型
print('将新特征转换为数值类型...')
# 使用LabelEncoder将组合特征转换为数值类型

# 先对训练集和测试集的组合特征进行编码
le = LabelEncoder()

# 对brand_model进行编码
combined_brand_model = pd.concat([train['brand_model'], test['brand_model']])
le.fit(combined_brand_model)
train['brand_model_enc'] = le.transform(train['brand_model'])
test['brand_model_enc'] = le.transform(test['brand_model'])

# 对brand_region进行编码
combined_brand_region = pd.concat([train['brand_region'], test['brand_region']])
le.fit(combined_brand_region)
train['brand_region_enc'] = le.transform(train['brand_region'])
test['brand_region_enc'] = le.transform(test['brand_region'])

# 对model_region进行编码
combined_model_region = pd.concat([train['model_region'], test['model_region']])
le.fit(combined_model_region)
train['model_region_enc'] = le.transform(train['model_region'])
test['model_region_enc'] = le.transform(test['model_region'])

print('添加高阶交互特征...')
# 添加高阶交互特征
train['power_squared'] = train['power'] ** 2
test['power_squared'] = test['power'] ** 2

train['power_log'] = np.log(train['power'] + 1)  # +1避免log(0)
test['power_log'] = np.log(test['power'] + 1)

train['km_log'] = np.log(train['km'] + 1)
test['km_log'] = np.log(test['km'] + 1)

train['car_age_log'] = np.log(train['car_age_days'] + 1)
test['car_age_log'] = np.log(test['car_age_days'] + 1)

# 添加更多统计特征
print('添加更多统计特征...')
# 品牌-地域平均价格
brand_region_avg_price = train.groupby(['brand', 'regionCode'])['price'].mean().to_dict()
train['brand_region_avg_price'] = train.set_index(['brand', 'regionCode']).index.map(brand_region_avg_price)
test['brand_region_avg_price'] = test.set_index(['brand', 'regionCode']).index.map(brand_region_avg_price)

# 品牌-地域平均功率
brand_region_avg_power = train.groupby(['brand', 'regionCode'])['power'].mean().to_dict()
train['brand_region_avg_power'] = train.set_index(['brand', 'regionCode']).index.map(brand_region_avg_power)
test['brand_region_avg_power'] = test.set_index(['brand', 'regionCode']).index.map(brand_region_avg_power)

# 型号-地域平均车龄
model_region_avg_age = train.groupby(['carModel', 'regionCode'])['car_age_days'].mean().to_dict()
train['model_region_avg_age'] = train.set_index(['carModel', 'regionCode']).index.map(model_region_avg_age)
test['model_region_avg_age'] = test.set_index(['carModel', 'regionCode']).index.map(model_region_avg_age)

# 添加更多交叉特征
print('添加更多交叉特征...')
train['power_brand_interaction'] = train['power'] * train['brand']
test['power_brand_interaction'] = test['power'] * test['brand']

train['km_region_interaction'] = train['km'] * train['regionCode']
test['km_region_interaction'] = test['km'] * test['regionCode']

train['car_age_brand_interaction'] = train['car_age_days'] * train['brand']
test['car_age_brand_interaction'] = test['car_age_days'] * test['brand']

# 添加新特征到feature_cols
new_features = ['power_km_ratio', 'power_age_ratio', 'km_age_ratio', 
                'brand_avg_power', 'region_avg_age', 'brand_model_avg_price',
                'brand_model_enc', 'brand_region_enc', 'model_region_enc',
                'power_squared', 'power_log', 'km_log', 'car_age_log',
                'brand_region_avg_price', 'brand_region_avg_power', 'model_region_avg_age',
                'power_brand_interaction', 'km_region_interaction', 'car_age_brand_interaction']
feature_cols.extend(new_features)

print(f'新增的特征组合: {new_features}')

# 清理临时列
train = train.drop(['brand_model', 'brand_region', 'model_region'], axis=1)
test = test.drop(['brand_model', 'brand_region', 'model_region'], axis=1)

## 提前特征列，标签列构造训练样本和测试样本
X_train = train[feature_cols]
y_train =train['price']

X_test = test[feature_cols]

print('\nX train shape:', X_train.shape)
print('X test shape:', X_test.shape)

# Cell 13
## 定义了一个统计函数，方便后续信息统计12

# 查看缺失值情况
print('缺失值情况:')
print(train[feature_cols].isnull().sum())

# 针对不同类型特征采用差异化的缺失值填充策略
# 分离离散型和连续型特征
# 虽然是float64类型，但从业务角度看是离散型特征
categorical_cols = []
continuous_cols = [col for col in feature_cols if col not in categorical_cols]

# 离散型特征使用众数填充（保持离散性质，适合One-Hot编码）
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_cat = cat_imputer.fit_transform(train[categorical_cols])
    X_test_cat = cat_imputer.transform(test[categorical_cols])
    print(f'离散型特征 {categorical_cols} 使用众数填充，众数为: {cat_imputer.statistics_}')

# 连续型特征使用均值填充（减少异常值影响）
if continuous_cols:
    cont_imputer = SimpleImputer(strategy='mean')
    X_train_cont = cont_imputer.fit_transform(train[continuous_cols])
    X_test_cont = cont_imputer.transform(test[continuous_cols])
    print(f'连续型特征 {continuous_cols} 使用均值填充，均值为: {cont_imputer.statistics_}')

# 合并特征
if categorical_cols and continuous_cols:
    X_train = np.hstack((X_train_cont, X_train_cat))
    X_test = np.hstack((X_test_cont, X_test_cat))
elif categorical_cols:
    X_train = X_train_cat
    X_test = X_test_cat
else:
    X_train = X_train_cont
    X_test = X_test_cont

print('\n填充后训练集形状:', X_train.shape)
print('填充后测试集形状:', X_test.shape)

# 2. 高级特征选择技术
print('\n2. 高级特征选择技术')

# 创建XGBoost模型对象
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# 2.1 互信息特征选择
print('\n2.1 互信息特征选择')
# 计算互信息
mi = mutual_info_regression(X_train, y_train)
mi_df = pd.DataFrame({'feature': feature_cols, 'mi_score': mi})
mi_df = mi_df.sort_values('mi_score', ascending=False)
print('\n互信息特征重要性排序（前20名）:')
print(mi_df.head(20))

# 2.2 基于XGBoost的特征重要性评估
temp_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
temp_model.fit(X_train, y_train)
importances = temp_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
print('\nXGBoost特征重要性排序（前20名）:')
print(feature_importance_df.head(20))

# 2.3 递归特征消除 (RFE)
print('\n2.3 递归特征消除 (RFE)')
# 使用线性回归作为基础模型进行RFE
rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=25, step=5, verbose=1)
rfe_selector.fit(X_train, y_train)
rfe_support = rfe_selector.support_

# 综合三种方法选择特征
# 1. 互信息前30个特征
mi_top_features = set(mi_df.head(30)['feature'])
# 2. XGBoost重要性前30个特征
xgb_top_features = set(feature_importance_df.head(30)['feature'])
# 3. RFE选择的特征
rfe_features = set([feature_cols[i] for i in range(len(rfe_support)) if rfe_support[i]])

# 取三种方法的交集和并集的平衡
# 先取三种方法都认可的特征
common_features = mi_top_features & xgb_top_features & rfe_features
# 再添加两种方法认可的特征
two_methods_features = (mi_top_features & xgb_top_features) | (mi_top_features & rfe_features) | (xgb_top_features & rfe_features)
# 最终特征集
final_feature_set = common_features.union(two_methods_features)

# 将特征名转换为索引
selected_features_idx = [feature_cols.index(f) for f in final_feature_set if f in feature_cols]

# 如果选择的特征数量过多，限制最大数量
max_features = 35
if len(selected_features_idx) > max_features:
    # 按XGBoost重要性排序并取前max_features个
    xgb_sorted_features = feature_importance_df.head(max_features)['feature'].tolist()
    selected_features_idx = [feature_cols.index(f) for f in xgb_sorted_features if f in feature_cols]

selected_features = [feature_cols[i] for i in selected_features_idx]
print(f'\n综合选择的特征数量: {len(selected_features)}')
print(f'综合选择的特征: {selected_features}')

# 更新训练集和测试集，只保留选择的特征
X_train_selected = X_train[:, selected_features_idx]
X_test_selected = X_test[:, selected_features_idx]

# 如果没有选择到特征，使用所有特征
if X_train_selected.shape[1] == 0:
    X_train_selected = X_train
    X_test_selected = X_test
    print('未选择到重要特征，使用所有特征')
else:
    X_train = X_train_selected
    X_test = X_test_selected
    print(f'特征选择后训练集形状: {X_train.shape}')
    print(f'特征选择后测试集形状: {X_test.shape}')

# 3. 数据增强技术
print('\n3. 数据增强技术')

# 生成更多的训练样本
print(f'原始训练集大小: {X_train.shape[0]}')

# 对训练数据进行重采样（增加数据多样性）
X_train_augmented, y_train_augmented = resample(X_train, y_train, 
                                                n_samples=int(X_train.shape[0] * 1.5), 
                                                random_state=42,
                                                replace=True)  # 允许重复采样

print(f'增强后训练集大小: {X_train_augmented.shape[0]}')

# 更新训练集为增强后的数据集
X_train = X_train_augmented
y_train = y_train_augmented

# Cell 15

# 5折交叉验证评估模型
print("开始5折交叉验证...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("交叉验证R2分数:", cv_scores)
print("平均R2分数:", cv_scores.mean())
print("R2分数标准差:", cv_scores.std())

# 可选：超参数网格搜索优化
## 3.1 模型调优 - 扩展超参数搜索范围
print('\n=== 模型调优 - 扩展超参数搜索范围 ===')

# 扩展超参数搜索空间 - 更全面的搜索范围
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  # 扩展迭代次数范围
    'learning_rate': [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],  # 增加学习率的精细粒度
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],  # 扩展深度范围
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],  # 增加子样本比例的选项
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],  # 增加列采样比例的选项
    'gamma': [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0],  # 扩展gamma范围
    'reg_alpha': [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],  # 扩展L1正则化范围
    'reg_lambda': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # 扩展L2正则化范围
}

# 使用RandomizedSearchCV进行超参数搜索 - 增加迭代次数
print('开始超参数随机搜索...')
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=200,  # 显著增加搜索的参数组合数量，从50增加到200
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print('最佳参数:', random_search.best_params_)
print('最佳交叉验证R2分数:', random_search.best_score_)

# Cell 17
## 4.1 实现早停策略
print('\n=== 实现早停策略 ===')

# 划分训练集和验证集
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 使用最佳参数并增加迭代次数，设置早停
# 复制最佳参数并更新n_estimators
best_params_with_early_stopping = random_search.best_params_.copy()
best_params_with_early_stopping['n_estimators'] = 1000  # 增加迭代次数

model_with_early_stopping = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    **best_params_with_early_stopping
)

print('开始训练带早停的模型...')
model_with_early_stopping.fit(
    X_train_part, y_train_part,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False  # 关闭训练过程输出
)

# 在训练集和验证集上评估模型
y_pred_train_part = model_with_early_stopping.predict(X_train_part)
y_pred_val = model_with_early_stopping.predict(X_val)
train_r2 = r2_score(y_train_part, y_pred_train_part)
val_r2 = r2_score(y_val, y_pred_val)
print(f'带早停的训练集R2分数: {train_r2}')
print(f'带早停的验证集R2分数: {val_r2}')
print(f'早停时的迭代次数: {model_with_early_stopping.best_iteration}')

# 使用带早停的模型进行最终预测
y_pred_train = model_with_early_stopping.predict(X_train)
y_pred_test = model_with_early_stopping.predict(X_test)
final_train_r2 = r2_score(y_train, y_pred_train)
print(f'最终训练集R2分数: {final_train_r2}')

# Cell 18 (Markdown)
# 4.2 模型集成

# Cell 18
print('\n=== 模型集成 ===')

# 训练多个不同参数的模型（包括LightGBM）
print('开始训练多个模型...')

# 模型1: XGBoost - 使用最佳参数
model1 = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    **random_search.best_params_
)
model1.fit(X_train, y_train)

# 模型2: XGBoost - 增加树深度，减少迭代次数
model2 = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=43
)
model2.fit(X_train, y_train)

# 模型3: XGBoost - 减少树深度，增加迭代次数
model3 = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=600,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=44
)
model3.fit(X_train, y_train)

# 模型4: XGBoost - 使用早停的模型（已训练）

# 模型5: LightGBM模型
print('训练LightGBM模型...')
model5 = LGBMRegressor(
    objective='regression',
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=45
)
model5.fit(X_train, y_train)

# 模型6: CatBoost模型
print('训练CatBoost模型...')
model6 = CatBoostRegressor(
    loss_function='RMSE',
    n_estimators=500,
    learning_rate=0.1,
    depth=6,
    subsample=0.9,
    colsample_bylevel=0.9,
    random_state=46,
    verbose=0  # 关闭训练过程输出
)
model6.fit(X_train, y_train)

# 模型7: RandomForest模型
print('训练RandomForest模型...')
model7 = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=47
)
model7.fit(X_train, y_train)

# 计算各模型的预测结果
print('开始融合预测结果...')
y_pred_test1 = model1.predict(X_test)
y_pred_test2 = model2.predict(X_test)
y_pred_test3 = model3.predict(X_test)
y_pred_test4 = y_pred_test  # 早停模型的预测结果
y_pred_test5 = model5.predict(X_test)  # LightGBM模型的预测结果
y_pred_test6 = model6.predict(X_test)  # CatBoost模型的预测结果
y_pred_test7 = model7.predict(X_test)  # RandomForest模型的预测结果

# 计算各模型的训练集R2分数作为权重
y_pred_train1 = model1.predict(X_train)
y_pred_train2 = model2.predict(X_train)
y_pred_train3 = model3.predict(X_train)
y_pred_train4 = y_pred_train  # 早停模型的预测结果
y_pred_train5 = model5.predict(X_train)  # LightGBM模型的预测结果
y_pred_train6 = model6.predict(X_train)  # CatBoost模型的预测结果
y_pred_train7 = model7.predict(X_train)  # RandomForest模型的预测结果

# 计算各模型的R2分数
r2_1 = r2_score(y_train, y_pred_train1)
r2_2 = r2_score(y_train, y_pred_train2)
r2_3 = r2_score(y_train, y_pred_train3)
r2_4 = r2_score(y_train, y_pred_train4)
r2_5 = r2_score(y_train, y_pred_train5)
r2_6 = r2_score(y_train, y_pred_train6)  # CatBoost模型的R2分数
r2_7 = r2_score(y_train, y_pred_train7)  # RandomForest模型的R2分数

# 显示各模型的R2分数
print(f'\n各模型R2分数：')
print(f'模型1 (XGB最佳参数): {r2_1:.4f}')
print(f'模型2 (XGB深度7): {r2_2:.4f}')
print(f'模型3 (XGB深度4): {r2_3:.4f}')
print(f'模型4 (XGB早停): {r2_4:.4f}')
print(f'模型5 (LightGBM): {r2_5:.4f}')
print(f'模型6 (CatBoost): {r2_6:.4f}')
print(f'模型7 (RandomForest): {r2_7:.4f}')

# 计算加权平均权重（使用R2分数作为权重）
total_r2 = r2_1 + r2_2 + r2_3 + r2_4 + r2_5 + r2_6 + r2_7
weight1 = r2_1 / total_r2
weight2 = r2_2 / total_r2
weight3 = r2_3 / total_r2
weight4 = r2_4 / total_r2
weight5 = r2_5 / total_r2
weight6 = r2_6 / total_r2  # CatBoost模型的权重
weight7 = r2_7 / total_r2  # RandomForest模型的权重

print(f'\n各模型权重：')
print(f'模型1权重: {weight1:.4f}')
print(f'模型2权重: {weight2:.4f}')
print(f'模型3权重: {weight3:.4f}')
print(f'模型4权重: {weight4:.4f}')
print(f'模型5权重: {weight5:.4f}')
print(f'模型6权重: {weight6:.4f}')
print(f'模型7权重: {weight7:.4f}')

# 使用加权平均融合
y_pred_ensemble = (y_pred_test1 * weight1 + 
                  y_pred_test2 * weight2 + 
                  y_pred_test3 * weight3 + 
                  y_pred_test4 * weight4 + 
                  y_pred_test5 * weight5 + 
                  y_pred_test6 * weight6 +  # CatBoost模型的预测结果
                  y_pred_test7 * weight7)  # RandomForest模型的预测结果

# 评估融合模型在训练集上的效果
y_pred_train_ensemble = (y_pred_train1 * weight1 + 
                        y_pred_train2 * weight2 + 
                        y_pred_train3 * weight3 + 
                        y_pred_train4 * weight4 + 
                        y_pred_train5 * weight5 + 
                        y_pred_train6 * weight6 +  # CatBoost模型的预测结果
                        y_pred_train7 * weight7)  # RandomForest模型的预测结果
ensemble_train_r2 = r2_score(y_train, y_pred_train_ensemble)
print(f'加权平均集成模型训练集R2分数: {ensemble_train_r2}')

# =================================================
# Stacking集成策略
# =================================================
print('\n=== 实现复杂集成策略 ===')

# 1. 基础Stacking策略
print('\n1. 基础Stacking策略')

# 准备基础模型列表
base_models = [
    ('xgb_best', model1),
    ('xgb_depth7', model2),
    ('xgb_depth4', model3),
    ('xgb_earlystop', model_with_early_stopping),
    ('lgbm', model5),
    ('catboost', model6),
    ('rf', model7)
]

# 创建Stacking集成模型，使用线性回归作为元模型
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5,  # 使用5折交叉验证
    passthrough=False  # 不将原始特征传递给元模型
)

# 训练Stacking模型
print('训练基础Stacking集成模型...')
stacking_model.fit(X_train, y_train)

# 预测Stacking集成结果
y_pred_train_stacking = stacking_model.predict(X_train)
y_pred_test_stacking = stacking_model.predict(X_test)

# 评估Stacking集成模型性能
stacking_train_r2 = r2_score(y_train, y_pred_train_stacking)
stacking_train_mae = mean_absolute_error(y_train, y_pred_train_stacking)
stacking_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_stacking))
print(f'基础Stacking集成模型训练集R2分数: {stacking_train_r2:.4f}')
print(f'基础Stacking集成模型训练集MAE: {stacking_train_mae:.4f}')
print(f'基础Stacking集成模型训练集RMSE: {stacking_train_rmse:.4f}')

# 2. 多层Stacking策略
print('\n2. 多层Stacking策略')

# 第一层Stacking：使用不同类型的模型作为基础模型
level0_models = [
    ('xgb_best', model1),
    ('xgb_depth7', model2),
    ('xgb_depth4', model3),
    ('xgb_earlystop', model_with_early_stopping),
    ('lgbm', model5),
    ('catboost', model6),
    ('rf', model7)
]

# 第二层Stacking：使用XGBoost作为元模型，接收第一层的输出
level1_models = [
    ('stacking1', StackingRegressor(
        estimators=level0_models,
        final_estimator=LinearRegression(),
        cv=5,
        passthrough=False
    )),
    # 添加额外的元模型
    ('xgb_meta', XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=50
    )),
    ('lgbm_meta', LGBMRegressor(
        objective='regression',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=51
    ))
]

# 最终层Stacking：将第二层的输出进行融合
final_stacking_model = StackingRegressor(
    estimators=level1_models,
    final_estimator=LinearRegression(),
    cv=5,
    passthrough=False
)

# 训练多层Stacking模型
print('训练多层Stacking集成模型...')
final_stacking_model.fit(X_train, y_train)

# 使用多层Stacking模型进行预测
y_pred_train_multi_stacking = final_stacking_model.predict(X_train)
y_pred_test_multi_stacking = final_stacking_model.predict(X_test)

# 评估多层Stacking模型性能
multi_stacking_train_r2 = r2_score(y_train, y_pred_train_multi_stacking)
multi_stacking_train_mae = mean_absolute_error(y_train, y_pred_train_multi_stacking)
multi_stacking_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_multi_stacking))
print(f'多层Stacking集成模型训练集R2分数: {multi_stacking_train_r2:.4f}')
print(f'多层Stacking集成模型训练集MAE: {multi_stacking_train_mae:.4f}')
print(f'多层Stacking集成模型训练集RMSE: {multi_stacking_train_rmse:.4f}')

# 3. 混合集成策略（Stacking + 加权平均）
print('\n3. 混合集成策略（Stacking + 加权平均）')

# 获取所有模型的预测结果
y_preds_all = [
    y_pred_train1, y_pred_train2, y_pred_train3, y_pred_train4,
    y_pred_train5, y_pred_train6, y_pred_train7, y_pred_train_ensemble,
    y_pred_train_stacking, y_pred_train_multi_stacking
]

# 计算每个模型的权重（基于R2分数）
r2_scores_all = [r2_score(y_train, pred) for pred in y_preds_all]
total_r2_all = sum(r2_scores_all)
weights_all = [r2 / total_r2_all for r2 in r2_scores_all]

# 计算混合集成的预测结果
y_pred_train_hybrid = np.zeros_like(y_train, dtype=np.float64)
for i, pred in enumerate(y_preds_all):
    y_pred_train_hybrid += pred * weights_all[i]

# 计算混合集成在测试集上的预测结果
y_preds_all_test = [
    y_pred_test1, y_pred_test2, y_pred_test3, y_pred_test4,
    y_pred_test5, y_pred_test6, y_pred_test7, y_pred_ensemble,
    y_pred_test_stacking, y_pred_test_multi_stacking
]

y_pred_test_hybrid = np.zeros_like(y_pred_test1, dtype=np.float64)
for i, pred in enumerate(y_preds_all_test):
    y_pred_test_hybrid += pred * weights_all[i]

# 评估混合集成模型性能
hybrid_train_r2 = r2_score(y_train, y_pred_train_hybrid)
hybrid_train_mae = mean_absolute_error(y_train, y_pred_train_hybrid)
hybrid_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_hybrid))
print(f'混合集成模型训练集R2分数: {hybrid_train_r2:.4f}')
print(f'混合集成模型训练集MAE: {hybrid_train_mae:.4f}')
print(f'混合集成模型训练集RMSE: {hybrid_train_rmse:.4f}')

# 4. 选择最佳集成模型
print('\n4. 选择最佳集成模型')
ensemble_models = [
    ('加权平均集成模型', y_pred_train_ensemble, y_pred_ensemble),
    ('基础Stacking集成模型', y_pred_train_stacking, y_pred_test_stacking),
    ('多层Stacking集成模型', y_pred_train_multi_stacking, y_pred_test_multi_stacking),
    ('混合集成模型', y_pred_train_hybrid, y_pred_test_hybrid)
]

best_model_name = ''
best_r2 = -float('inf')
best_y_pred_test = None

for model_name, y_pred_train, y_pred_test in ensemble_models:
    current_r2 = r2_score(y_train, y_pred_train)
    if current_r2 > best_r2:
        best_r2 = current_r2
        best_model_name = model_name
        best_y_pred_test = y_pred_test

print(f'最佳集成模型: {best_model_name}, R2分数: {best_r2:.4f}')

# Cell 18 (Markdown)
# 5. 评估与分析优化

# Cell 19
print('\n=== 5.1 误差分析 ===')

# 计算残差
train_with_predictions = train.copy()
train_with_predictions['pred_price'] = y_pred_train_ensemble
train_with_predictions['residual'] = train_with_predictions['price'] - train_with_predictions['pred_price']
train_with_predictions['abs_residual'] = train_with_predictions['residual'].abs()

# 分析残差与重要特征的关系
print('分析残差与重要特征的关系...')

# 残差与功率的关系
plt.figure(figsize=(12, 6))
plt.scatter(train_with_predictions['power'], train_with_predictions['abs_residual'], alpha=0.5)
plt.xlabel('功率')
plt.ylabel('绝对残差')
plt.title('功率与预测残差的关系')
plt.savefig('residual_vs_power.png')
print('残差与功率的关系图已保存为 residual_vs_power.png')

# 残差与里程的关系
plt.figure(figsize=(12, 6))
plt.scatter(train_with_predictions['km'], train_with_predictions['abs_residual'], alpha=0.5)
plt.xlabel('里程')
plt.ylabel('绝对残差')
plt.title('里程与预测残差的关系')
plt.savefig('residual_vs_km.png')
print('残差与里程的关系图已保存为 residual_vs_km.png')

# 残差与车龄的关系
plt.figure(figsize=(12, 6))
plt.scatter(train_with_predictions['car_age_days'], train_with_predictions['abs_residual'], alpha=0.5)
plt.xlabel('车龄（天）')
plt.ylabel('绝对残差')
plt.title('车龄与预测残差的关系')
plt.savefig('residual_vs_car_age.png')
print('残差与车龄的关系图已保存为 residual_vs_car_age.png')

# 查看误差最大的样本
top_error_samples = train_with_predictions.sort_values('abs_residual', ascending=False).head(20)
print("\n误差最大的样本：")
print(top_error_samples[['price', 'pred_price', 'residual', 'abs_residual', 'power', 'km', 'car_age_days']])

# Cell 20
print('\n=== 5.2 多指标评估 ===')

# 计算多种评估指标
r2 = ensemble_train_r2
mae = mean_absolute_error(y_train, y_pred_train_ensemble)
mse = mean_squared_error(y_train, y_pred_train_ensemble)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_train - y_pred_train_ensemble) / y_train)) * 100

print(f"集成模型性能评估：")
print(f"R2分数: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# 对比各个模型的性能
print('\n各模型性能对比：')

# 对比各模型性能
models = [
    ('最佳参数模型', y_pred_train1),
    ('深度7模型', y_pred_train2),
    ('深度4模型', y_pred_train3),
    ('早停模型', y_pred_train4),
    ('LightGBM模型', y_pred_train5),
    ('CatBoost模型', y_pred_train6),
    ('RandomForest模型', y_pred_train7),
    ('加权平均集成模型', y_pred_train_ensemble),
    ('Stacking集成模型', y_pred_train_stacking)
]

for model_name, y_pred in models:
    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    print(f"{model_name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

# 生成最终提交文件
out_df = pd.DataFrame()
out_df['ID'] = test['ID']
out_df['price'] = y_pred_stacking  # 使用Stacking集成的结果提交

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
