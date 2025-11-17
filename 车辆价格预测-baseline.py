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
numerical_cols = train.select_dtypes(exclude = 'object').columns
print(numerical_cols)

# Cell 11
## 选择特征列 - 包含新的时间特征和Type特征
feature_cols = [
    col for col in numerical_cols if col not in [
        'ID', 'name', 'regDate', 'creatDate', 'price', 'model', 'brand',
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
## 2.3 特征组合与特征选择
print('\n=== 特征组合与特征选择 ===')

# 创建特征组合
print('\n1. 创建特征组合')
train['power_km_ratio'] = train['power'] / (train['km'] + 1)  # +1避免除零
test['power_km_ratio'] = test['power'] / (test['km'] + 1)

# 创建其他可能的特征组合
train['power_age_ratio'] = train['power'] / (train['car_age_days'] + 1)
test['power_age_ratio'] = test['power'] / (test['car_age_days'] + 1)

train['km_age_ratio'] = train['km'] / (train['car_age_days'] + 1)
test['km_age_ratio'] = test['km'] / (test['car_age_days'] + 1)

print(f'新增的特征组合: power_km_ratio, power_age_ratio, km_age_ratio')

# 更新特征列列表
feature_cols.extend(['power_km_ratio', 'power_age_ratio', 'km_age_ratio'])

## 提前特征列，标签列构造训练样本和测试样本
X_train = train[feature_cols]
y_train =train['price']

X_test = test[feature_cols]

print('\nX train shape:', X_train.shape)
print('X test shape:', X_test.shape)

# Cell 13
## 定义了一个统计函数，方便后续信息统计12
from sklearn.impute import SimpleImputer
import numpy as np

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

# 2. 基于XGBoost的特征重要性评估和选择
print('\n2. 特征重要性评估与选择')

# 导入必要的库
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

# 创建XGBoost模型对象
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# 先训练一个基础模型用于特征重要性评估
temp_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
temp_model.fit(X_train, y_train)

# 获取特征重要性
importances = temp_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

print('\n特征重要性排序（前20名）:')
print(feature_importance_df.head(20))

# 选择重要特征（重要性大于0的特征）
selected_features_idx = np.where(importances > 0)[0]

# 如果选择的特征数量过多，限制最大数量
max_features = 30
if len(selected_features_idx) > max_features:
    selected_features_idx = selected_features_idx[:max_features]

selected_features = [feature_cols[i] for i in selected_features_idx]
print(f'\n选择的特征数量: {len(selected_features)}')
print(f'选择的特征: {selected_features}')

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

# 使用RandomizedSearchCV代替GridSearchCV进行更全面的超参数搜索
from sklearn.model_selection import RandomizedSearchCV

# 扩展超参数搜索空间
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 9, 11],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0, 2.0]
}

# 使用RandomizedSearchCV进行超参数搜索
print('开始超参数随机搜索...')
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # 搜索的参数组合数量
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

from sklearn.model_selection import train_test_split

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

# 训练多个不同参数的XGBoost模型
print('开始训练多个模型...')

# 模型1: 使用最佳参数
model1 = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    **random_search.best_params_
)
model1.fit(X_train, y_train)

# 模型2: 增加树深度，减少迭代次数
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

# 模型3: 减少树深度，增加迭代次数
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

# 模型4: 使用早停的模型（已训练）

# 融合预测结果
print('开始融合预测结果...')
y_pred_test1 = model1.predict(X_test)
y_pred_test2 = model2.predict(X_test)
y_pred_test3 = model3.predict(X_test)
y_pred_test4 = y_pred_test  # 早停模型的预测结果

# 使用简单平均融合
y_pred_ensemble = (y_pred_test1 + y_pred_test2 + y_pred_test3 + y_pred_test4) / 4

# 评估融合模型在训练集上的效果
y_pred_train1 = model1.predict(X_train)
y_pred_train2 = model2.predict(X_train)
y_pred_train3 = model3.predict(X_train)
y_pred_train4 = y_pred_train  # 早停模型的预测结果
y_pred_train_ensemble = (y_pred_train1 + y_pred_train2 + y_pred_train3 + y_pred_train4) / 4
ensemble_train_r2 = r2_score(y_train, y_pred_train_ensemble)
print(f'集成模型训练集R2分数: {ensemble_train_r2}')

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

# 保存残差分布图
plt.figure(figsize=(12, 6))
plt.scatter(train_with_predictions['power'], train_with_predictions['abs_residual'], alpha=0.5)
plt.xlabel('功率')
plt.ylabel('绝对残差')
plt.title('功率与预测残差的关系')
plt.savefig('residual_vs_power.png')
print('残差与功率的关系图已保存为 residual_vs_power.png')

plt.figure(figsize=(12, 6))
plt.scatter(train_with_predictions['km'], train_with_predictions['abs_residual'], alpha=0.5)
plt.xlabel('里程')
plt.ylabel('绝对残差')
plt.title('里程与预测残差的关系')
plt.savefig('residual_vs_km.png')
print('残差与里程的关系图已保存为 residual_vs_km.png')

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

# 导入所需的评估指标
from sklearn.metrics import mean_absolute_error

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

# 计算单个模型的性能
models = [
    ('最佳参数模型', y_pred_train1),
    ('深度7模型', y_pred_train2),
    ('深度4模型', y_pred_train3),
    ('早停模型', y_pred_train4),
    ('集成模型', y_pred_train_ensemble)
]

for model_name, y_pred in models:
    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    print(f"{model_name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

# Cell 15
out_df = pd.DataFrame()
out_df['ID'] = test['ID']
out_df['price'] = y_pred_ensemble

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
