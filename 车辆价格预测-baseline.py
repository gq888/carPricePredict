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
# | price | 二手汽车交易价格（预测目标y） |
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
import seaborn as sns
import datetime
import pickle
import os
import warnings

# 设置警告过滤
warnings.filterwarnings('ignore')

# 导入模型相关库
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBRegressor

# 导入评估和选择相关库
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectKBest, RFE
from sklearn.utils import resample

# 定义常量
STUDENT_ID = '25451354008'
DATA_DIR = 'data/'
MODEL_DIR = ''
RANDOM_SEED = 42

# 文件路径常量
RANDOM_SEARCH_FILE = 'random_search_results.pkl'
XGB_EARLYSTOP_FILE = 'xgb_earlystop_model.pkl'
XGB_MODEL1_FILE = 'xgb_model1.pkl'
XGB_MODEL2_FILE = 'xgb_model2.pkl'
XGB_MODEL3_FILE = 'xgb_model3.pkl'
CATBOOST_MODEL_FILE = 'catboost_model.pkl'
RANDOM_FOREST_MODEL_FILE = 'random_forest_model.pkl'
BASE_STACKING_FILE = 'base_stacking_model.pkl'
MULTI_STACKING_FILE = 'multi_stacking_model.pkl'
FEATURE_SELECTION_FILE = 'feature_selection_results.pkl'
PREPROCESSING_FILE = 'preprocessing_params.pkl'
DATA_PIPELINE_FILE = 'data_pipeline_results.pkl'

# 定义函数：模型持久化
def save_model(model, file_path):
    """保存模型到文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'模型已保存到 {file_path}')

def load_model(file_path):
    """从文件加载模型"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f'已从 {file_path} 加载模型')
        return model
    return None

def save_feature_selection(selected_features, selected_features_idx, file_path):
    """保存特征选择结果"""
    with open(file_path, 'wb') as f:
        pickle.dump((selected_features, selected_features_idx), f)
    print(f'特征选择结果已保存到 {file_path}')

def load_feature_selection(file_path):
    """从文件加载特征选择结果"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            selected_features, selected_features_idx = pickle.load(f)
        print(f'已从 {file_path} 加载特征选择结果')
        return selected_features, selected_features_idx
    return None, None

def save_preprocessing_params(cat_imputer, cont_imputer, file_path):
    """保存预处理参数"""
    with open(file_path, 'wb') as f:
        pickle.dump((cat_imputer, cont_imputer), f)
    print(f'预处理参数已保存到 {file_path}')

def load_preprocessing_params(file_path):
    """从文件加载预处理参数"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            cat_imputer, cont_imputer = pickle.load(f)
        print(f'已从 {file_path} 加载预处理参数')
        return cat_imputer, cont_imputer
    return None, None

def get_or_train_model(model, file_path, X_train, y_train, **fit_params):
    """获取已保存的模型或训练新模型"""
    loaded_model = load_model(file_path)
    if loaded_model is not None:
        return loaded_model
    
    print(f'开始训练模型...')
    model.fit(X_train, y_train, **fit_params)
    save_model(model, file_path)
    return model

def save_full_pipeline(X_train, y_train, X_test, feature_cols, test_ids, file_path):
    """保存完整的数据处理管道结果"""
    with open(file_path, 'wb') as f:
        pickle.dump((X_train, y_train, X_test, feature_cols, test_ids), f)
    print(f'完整数据处理管道结果已保存到 {file_path}')

def load_full_pipeline(file_path):
    """从文件加载完整的数据处理管道结果"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if len(data) == 5:
            X_train, y_train, X_test, feature_cols, test_ids = data
            print(f'已从 {file_path} 加载完整数据处理管道结果，包含测试数据ID')
            return X_train, y_train, X_test, feature_cols, test_ids
        else:
            # 兼容旧版本的保存格式
            X_train, y_train, X_test, feature_cols = data
            print(f'已从 {file_path} 加载完整数据处理管道结果（旧版本格式）')
            return X_train, y_train, X_test, feature_cols, None
    return None, None, None, None, None
    
# 读取数据
print('\n1. 读取数据')
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'testA.csv'))

# 定义函数：数据读取和预处理
def load_and_preprocess_data():
    """读取并预处理训练和测试数据"""
    print('=== 数据读取和预处理 ===')
    
    # 输出数据的大小信息
    print('Train data shape:', train.shape)
    print('TestA data shape:', test.shape)
    
    # 时间特征提取
    print('\n2. 时间特征提取')
    
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
    for df in [train, test]:
        for col in ['regDate', 'creatDate']:
            df[f'{col}_str'] = df[col].astype(str)
            df[col] = df[f'{col}_str'].apply(process_date)
        
        # 清理临时字符串列
        df.drop([f'regDate_str', f'creatDate_str'], axis=1, inplace=True)
        
        # 提取时间特征
        df['reg_year'] = df['regDate'].dt.year
        df['reg_month'] = df['regDate'].dt.month
        df['reg_quarter'] = df['regDate'].dt.quarter
        
        df['creat_year'] = df['creatDate'].dt.year
        df['creat_month'] = df['creatDate'].dt.month
        df['creat_quarter'] = df['creatDate'].dt.quarter
        df['creat_dayofweek'] = df['creatDate'].dt.dayofweek
        
        # 计算车辆使用年限（天）
        df['car_age_days'] = (df['creatDate'] - df['regDate']).dt.days
    
    # 查看转换结果
    print(f"regDate转换成功: {train['regDate'].notna().sum()}/{len(train)}")
    print(f"creatDate转换成功: {train['creatDate'].notna().sum()}/{len(train)}")
    
    # 查看新添加的时间特征
    time_features = ['reg_year', 'reg_month', 'reg_quarter', 'creat_year', 'creat_month', 'creat_quarter', 'creat_dayofweek', 'car_age_days']
    print('添加的时间特征:', time_features)
    print('时间特征示例:')
    print(train[time_features].head())
    
    # 查看原始数据中的品牌、型号和地域列信息
    print('\n3. 数据探索')
    print('原始数据中的品牌、型号和地域列信息：')
    print('品牌(brand) - 唯一值数量:', train['brand'].nunique())
    print('型号(carModel) - 唯一值数量:', train['carModel'].nunique())
    print('地域(regionCode) - 唯一值数量:', train['regionCode'].nunique())
    
    # 查看这些列的一些基本统计信息
    print('\n品牌、型号和地域的基本统计信息：')
    print('品牌分布：', train['brand'].value_counts().head(10))
    print('型号分布：', train['carModel'].value_counts().head(10))
    print('地域分布：', train['regionCode'].value_counts().head(10))
    
    # 选择特征列
    numerical_cols = train.select_dtypes(exclude = 'object').columns
    print('\n数值列：', numerical_cols)
    
    feature_cols = [
        col for col in numerical_cols if col not in [
            'ID', 'name', 'regDate', 'creatDate', 'price', 'carModel', 'brand',
            'regionCode', 'seller'
        ]
    ]
    # 重新纳入Type特征
    feature_cols.extend(['BType', 'FType'])
    
    # 异常值处理
    print('\n4. 异常值处理')
    # power特征存在严重异常值，进行处理
    print('power特征异常值处理前:')
    print(f'power最大值: {train["power"].max()}')
    print(f'power大于600的样本数: {len(train[train["power"] > 600])}')
    
    # 使用99.9%分位数截断power异常值
    power_upper = train['power'].quantile(0.999)
    train.loc[train['power'] > power_upper, 'power'] = power_upper
    test.loc[test['power'] > power_upper, 'power'] = power_upper
    
    print('\npower特征异常值处理后:')
    print(f'power最大值: {train["power"].max()}')
    print(f'power大于600的样本数: {len(train[train["power"] > 600])}')
    print(f'使用的截断值: {power_upper}')
    
    return train, test, feature_cols

# 定义函数：特征工程
def perform_feature_engineering(train, test, feature_cols):
    """执行特征工程"""
    print('\n=== 特征组合与特征选择 ===')
    
    # 1. 创建特征组合
    print('\n1. 创建特征组合')
    
    # 创建比率特征
    print('创建比率特征...')
    for df in [train, test]:
        df['power_km_ratio'] = df['power'] / (df['km'] + 1)  # +1避免除零
        df['power_age_ratio'] = df['power'] / (df['car_age_days'] + 1)
        df['km_age_ratio'] = df['km'] / (df['car_age_days'] + 1)
    
    # 基于品牌、型号和地域的组合特征
    print('\n创建基于品牌、型号和地域的组合特征...')
    
    # 1. 品牌-型号组合
    print('创建品牌-型号组合特征...')
    for df in [train, test]:
        df['brand_model'] = df['brand'].astype(str) + '_' + df['carModel'].astype(str)
    
    # 2. 品牌-地域组合
    print('创建品牌-地域组合特征...')
    for df in [train, test]:
        df['brand_region'] = df['brand'].astype(str) + '_' + df['regionCode'].astype(str)
    
    # 3. 型号-地域组合
    print('创建型号-地域组合特征...')
    for df in [train, test]:
        df['model_region'] = df['carModel'].astype(str) + '_' + df['regionCode'].astype(str)
    
    # 4. 品牌平均功率特征
    print('创建品牌平均功率特征...')
    brand_avg_power = train.groupby('brand')['power'].mean().to_dict()
    for df in [train, test]:
        df['brand_avg_power'] = df['brand'].map(brand_avg_power)
    
    # 5. 地域平均车龄特征
    print('创建地域平均车龄特征...')
    region_avg_age = train.groupby('regionCode')['car_age_days'].mean().to_dict()
    for df in [train, test]:
        df['region_avg_age'] = df['regionCode'].map(region_avg_age)
    
    # 6. 品牌-型号平均价格特征（用于参考，不直接作为特征）
    print('创建品牌-型号平均价格特征...')
    brand_model_avg_price = train.groupby(['brand', 'carModel'])['price'].mean().to_dict()
    for df in [train, test]:
        df['brand_model_avg_price'] = df.set_index(['brand', 'carModel']).index.map(brand_model_avg_price)
    
    # 将新特征转换为数值类型
    print('将新特征转换为数值类型...')
    # 使用LabelEncoder将组合特征转换为数值类型
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
    for df in [train, test]:
        df['power_squared'] = df['power'] ** 2
        df['power_log'] = np.log(df['power'] + 1)  # +1避免log(0)
        df['km_log'] = np.log(df['km'] + 1)
        df['car_age_log'] = np.log(df['car_age_days'] + 1)
    
    # 添加更多统计特征
    print('添加更多统计特征...')
    # 品牌-地域平均价格
    brand_region_avg_price = train.groupby(['brand', 'regionCode'])['price'].mean().to_dict()
    for df in [train, test]:
        df['brand_region_avg_price'] = df.set_index(['brand', 'regionCode']).index.map(brand_region_avg_price)
    
    # 品牌-地域平均功率
    brand_region_avg_power = train.groupby(['brand', 'regionCode'])['power'].mean().to_dict()
    for df in [train, test]:
        df['brand_region_avg_power'] = df.set_index(['brand', 'regionCode']).index.map(brand_region_avg_power)
    
    # 型号-地域平均车龄
    model_region_avg_age = train.groupby(['carModel', 'regionCode'])['car_age_days'].mean().to_dict()
    for df in [train, test]:
        df['model_region_avg_age'] = df.set_index(['carModel', 'regionCode']).index.map(model_region_avg_age)
    
    # 添加更多交叉特征
    print('添加更多交叉特征...')
    for df in [train, test]:
        df['power_brand_interaction'] = df['power'] * df['brand']
        df['km_region_interaction'] = df['km'] * df['regionCode']
        df['car_age_brand_interaction'] = df['car_age_days'] * df['brand']
    
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
    
    # 提前特征列，标签列构造训练样本和测试样本
    X_train = train[feature_cols]
    y_train = train['price']
    X_test = test[feature_cols]
    
    print('\n数据形状:')
    print('X train shape:', X_train.shape)
    print('X test shape:', X_test.shape)
    print('y train shape:', y_train.shape)
    
    return X_train, y_train, X_test, feature_cols

# 首先检查是否有保存的完整数据集
print('检查是否有保存的完整数据集...')
skip_data_processing = False
X_train = None
X_test = None
y_train = None
feature_cols = None
test_ids = None

if os.path.exists(DATA_PIPELINE_FILE):
    print('发现已保存的完整数据集，尝试加载...')
    loaded_X_train, loaded_y_train, loaded_X_test, loaded_feature_cols, loaded_test_ids = load_full_pipeline(DATA_PIPELINE_FILE)

    if loaded_X_train is not None and loaded_y_train is not None and loaded_X_test is not None:
        print('成功加载完整数据集！')
        X_train = loaded_X_train
        y_train = loaded_y_train
        X_test = loaded_X_test
        feature_cols = loaded_feature_cols
        test_ids = loaded_test_ids
        skip_data_processing = True
        print(f'已加载数据集，训练集形状: {X_train.shape}')
        print(f'已加载数据集，测试集形状: {X_test.shape}')
        print(f'已加载数据集，特征数量: {len(feature_cols)}')
        if test_ids is not None:
            print(f'已加载测试数据ID，数量: {len(test_ids)}')
    else:
        print('加载的数据集不完整，将重新处理数据...')
        skip_data_processing = False
else:
    print('未发现保存的完整数据集，将执行完整数据处理流程...')
    skip_data_processing = False

# 如果没有跳过数据处理，则执行完整的数据处理流程
if not skip_data_processing:
    print('执行完整数据处理流程...')
    # 调用函数执行数据处理流程
    train, test, feature_cols = load_and_preprocess_data()
    X_train, y_train, X_test, feature_cols = perform_feature_engineering(train, test, feature_cols)

    # 转换为numpy数组以进行后续处理
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values

# 如果没有跳过数据处理，则执行特征选择
if not skip_data_processing:
    print('缺失值情况:')
    print(train[feature_cols].isnull().sum())

    # 针对不同类型特征采用差异化的缺失值填充策略
    # 分离离散型和连续型特征
    # 虽然是float64类型，但从业务角度看是离散型特征
    categorical_cols = []
    continuous_cols = [col for col in feature_cols if col not in categorical_cols]

    # 加载预处理参数或创建新的
    cat_imputer, cont_imputer = load_preprocessing_params(PREPROCESSING_FILE)

    if cat_imputer is None or cont_imputer is None:
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

        # 保存预处理参数
        save_preprocessing_params(cat_imputer, cont_imputer, PREPROCESSING_FILE)
    else:
        # 使用加载的预处理参数
        if categorical_cols:
            X_train_cat = cat_imputer.transform(train[categorical_cols])
            X_test_cat = cat_imputer.transform(test[categorical_cols])
            print(f'使用已加载的预处理参数：离散型特征 {categorical_cols} 使用众数填充')

        if continuous_cols:
            X_train_cont = cont_imputer.transform(train[continuous_cols])
            X_test_cont = cont_imputer.transform(test[continuous_cols])
            print(f'使用已加载的预处理参数：连续型特征 {continuous_cols} 使用均值填充')

    # 合并特征
    if categorical_cols and continuous_cols:
        X_train = np.hstack((X_train_cont, X_train_cat))
        X_test = np.hstack((X_test_cont, X_test_cat))
    elif categorical_cols:
        X_train = X_train_cat
        X_test = X_test_cat
    elif continuous_cols:
        X_train = X_train_cont
        X_test = X_test_cont
    else:
        raise ValueError("没有可用于训练的特征")

    print('\n填充后训练集形状:', X_train.shape)
    print('填充后测试集形状:', X_test.shape)

# 2. 高级特征选择技术
print('\n2. 高级特征选择技术')

# 如果没有跳过数据处理，则执行特征选择
if not skip_data_processing:
    # 尝试加载已保存的特征选择结果
    selected_features, selected_features_idx = load_feature_selection(FEATURE_SELECTION_FILE)

    if selected_features is None or selected_features_idx is None:
        print('未找到已保存的特征选择结果，将执行特征选择...')
        
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
        
        # 保存特征选择结果
        save_feature_selection(selected_features, selected_features_idx, FEATURE_SELECTION_FILE)
    else:
        print(f'已加载特征选择结果，共选择了 {len(selected_features)} 个特征:')
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
else:
    print('已加载完整数据集，跳过特征选择步骤...')
    selected_features = feature_cols  # 使用加载数据中的特征列

# 3. 数据增强技术
print('\n3. 数据增强技术')

# 原始训练集大小
print(f'原始训练集大小: {X_train.shape[0]}')

# 检查是否需要进行数据增强（当加载模型时不需要重新增强数据）
# 我们只在首次训练模型时进行数据增强
if not os.path.exists(XGB_EARLYSTOP_FILE) or not os.path.exists(XGB_MODEL1_FILE):
    # 使用重采样进行数据增强，将训练集大小增加50%
    X_train_augmented, y_train_augmented = resample(X_train, y_train, 
                                                    n_samples=int(X_train.shape[0] * 1.5), 
                                                    random_state=RANDOM_SEED,
                                                    replace=True)  # 允许重复采样

    print(f'增强后训练集大小: {X_train_augmented.shape[0]}')

    # 更新训练集为增强后的数据集
    X_train = X_train_augmented
    y_train = y_train_augmented
    # 保存完整的数据处理管道结果
    # 保存经过完整数据处理（包括特征选择）后的数据
    save_full_pipeline(X_train, y_train, X_test, selected_features, test['ID'].values, DATA_PIPELINE_FILE)
else:
    print('使用已保存的模型，跳过数据增强步骤')
    # 加载完整的数据处理管道结果
    loaded_X_train, loaded_y_train, loaded_X_test, loaded_feature_cols, loaded_test_ids = load_full_pipeline(DATA_PIPELINE_FILE)
    if loaded_X_train is not None:
        X_train = loaded_X_train
        y_train = loaded_y_train
        X_test = loaded_X_test
        feature_cols = loaded_feature_cols
        test_ids = loaded_test_ids
        print(f'已加载完整的数据处理管道结果，训练集形状: {X_train.shape}')
        print(f'已加载完整的数据处理管道结果，测试集形状: {X_test.shape}')
        print(f'已加载完整的数据处理管道结果，特征数量: {len(feature_cols)}')
        if test_ids is not None:
            print(f'已加载测试数据ID，数量: {len(test_ids)}')

# Cell 15

# 定义基础模型进行5折交叉验证
base_model = XGBRegressor(objective='reg:squarederror', random_state=RANDOM_SEED)

# 5折交叉验证评估模型
print("开始5折交叉验证...")
cv_scores = cross_val_score(base_model, X_train, y_train, cv=5, scoring='r2')
print("交叉验证R2分数:", cv_scores)
print("平均R2分数:", cv_scores.mean())
print("R2分数标准差:", cv_scores.std())

# 使用基础模型作为后续搜索的起始点
model = base_model

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
random_search_file = 'random_search_results.pkl'
if os.path.exists(random_search_file):
    print('加载已保存的随机搜索结果...')
    with open(random_search_file, 'rb') as f:
        random_search = pickle.load(f)
else:
    print('开始超参数随机搜索...')
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=200,  # 显著增加搜索的参数组合数量，从50增加到200
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    random_search.fit(X_train, y_train)
    
    # 保存随机搜索结果
    print('保存随机搜索结果...')
    with open(random_search_file, 'wb') as f:
        pickle.dump(random_search, f)

# 输出最佳参数和最佳得分
print('最佳参数:', random_search.best_params_)
print('最佳交叉验证R2分数:', random_search.best_score_)

# Cell 17
## 4.1 实现早停策略
print('\n=== 实现早停策略 ===')

# 划分训练集和验证集
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
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

# 使用get_or_train_model函数处理早停模型
model_with_early_stopping = get_or_train_model(model_with_early_stopping, XGB_EARLYSTOP_FILE, X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

# Cell 18 (Markdown)
# 4.2 模型集成

# Cell 18
print('\n=== 模型集成 ===')

# 训练多个不同参数的模型
print('开始训练多个模型...')

# 模型1: XGBoost - 使用最佳参数
model1 = XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_SEED,
    **random_search.best_params_
)
model1 = get_or_train_model(model1, XGB_MODEL1_FILE, X_train, y_train)

# 模型2: XGBoost - 增加树深度，减少迭代次数
model2 = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_SEED + 1
)
model2 = get_or_train_model(model2, XGB_MODEL2_FILE, X_train, y_train)

# 模型3: XGBoost - 减少树深度，增加迭代次数
model3 = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=600,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED + 2
)
model3 = get_or_train_model(model3, XGB_MODEL3_FILE, X_train, y_train)

# 模型4: XGBoost - 使用早停的模型（已训练）



# 模型6: CatBoost模型
model6 = CatBoostRegressor(
    loss_function='RMSE',
    n_estimators=500,
    learning_rate=0.1,
    depth=6,
    subsample=0.9,
    colsample_bylevel=0.9,
    random_state=RANDOM_SEED + 4,
    verbose=0  # 关闭训练过程输出
)
model6 = get_or_train_model(model6, CATBOOST_MODEL_FILE, X_train, y_train)

# 模型7: RandomForest模型
model7 = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=RANDOM_SEED + 5
)
model7 = get_or_train_model(model7, RANDOM_FOREST_MODEL_FILE, X_train, y_train)

# 计算各模型的预测结果
print('开始融合预测结果...')
y_pred_test1 = model1.predict(X_test)
y_pred_test2 = model2.predict(X_test)
y_pred_test3 = model3.predict(X_test)
y_pred_test4 = model_with_early_stopping.predict(X_test)  # 早停模型的预测结果
y_pred_test6 = model6.predict(X_test)  # CatBoost模型的预测结果
y_pred_test7 = model7.predict(X_test)  # RandomForest模型的预测结果

# 计算各模型的训练集R2分数作为权重
y_pred_train1 = model1.predict(X_train)
y_pred_train2 = model2.predict(X_train)
y_pred_train3 = model3.predict(X_train)
y_pred_train4 = model_with_early_stopping.predict(X_train)  # 早停模型的预测结果
y_pred_train6 = model6.predict(X_train)  # CatBoost模型的预测结果
y_pred_train7 = model7.predict(X_train)  # RandomForest模型的预测结果

# 计算各模型的R2分数
r2_1 = r2_score(y_train, y_pred_train1)
r2_2 = r2_score(y_train, y_pred_train2)
r2_3 = r2_score(y_train, y_pred_train3)
r2_4 = r2_score(y_train, y_pred_train4)
r2_6 = r2_score(y_train, y_pred_train6)  # CatBoost模型的R2分数
r2_7 = r2_score(y_train, y_pred_train7)  # RandomForest模型的R2分数

# 显示各模型的R2分数
print(f'\n各模型R2分数：')
print(f'模型1 (XGB最佳参数): {r2_1:.4f}')
print(f'模型2 (XGB深度7): {r2_2:.4f}')
print(f'模型3 (XGB深度4): {r2_3:.4f}')
print(f'模型4 (XGB早停): {r2_4:.4f}')

print(f'模型6 (CatBoost): {r2_6:.4f}')
print(f'模型7 (RandomForest): {r2_7:.4f}')

# 计算加权平均权重（使用R2分数作为权重）
total_r2 = r2_1 + r2_2 + r2_3 + r2_4 + r2_6 + r2_7
weight1 = r2_1 / total_r2
weight2 = r2_2 / total_r2
weight3 = r2_3 / total_r2
weight4 = r2_4 / total_r2
weight6 = r2_6 / total_r2  # CatBoost模型的权重
weight7 = r2_7 / total_r2  # RandomForest模型的权重

print(f'\n各模型权重：')
print(f'模型1权重: {weight1:.4f}')
print(f'模型2权重: {weight2:.4f}')
print(f'模型3权重: {weight3:.4f}')
print(f'模型4权重: {weight4:.4f}')

print(f'模型6权重: {weight6:.4f}')
print(f'模型7权重: {weight7:.4f}')

# 使用加权平均融合
y_pred_ensemble = (y_pred_test1 * weight1 + 
                  y_pred_test2 * weight2 + 
                  y_pred_test3 * weight3 + 
                  y_pred_test4 * weight4 + 
                  y_pred_test6 * weight6 +  # CatBoost模型的预测结果
                  y_pred_test7 * weight7)  # RandomForest模型的预测结果

# 评估融合模型在训练集上的效果
y_pred_train_ensemble = (y_pred_train1 * weight1 + 
                        y_pred_train2 * weight2 + 
                        y_pred_train3 * weight3 + 
                        y_pred_train4 * weight4 + 
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
stacking_model = get_or_train_model(stacking_model, BASE_STACKING_FILE, X_train, y_train)

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
final_stacking_model = get_or_train_model(final_stacking_model, MULTI_STACKING_FILE, X_train, y_train)

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
    y_pred_train6, y_pred_train7, y_pred_train_ensemble,
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
    y_pred_test6, y_pred_test7, y_pred_ensemble,
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
# print('\n=== 5.1 误差分析 ===')

# # 计算残差
# train_with_predictions = train.copy()
# train_with_predictions['pred_price'] = y_pred_train_ensemble
# train_with_predictions['residual'] = train_with_predictions['price'] - train_with_predictions['pred_price']
# train_with_predictions['abs_residual'] = train_with_predictions['residual'].abs()

# # 分析残差与重要特征的关系
# print('分析残差与重要特征的关系...')

# # 残差与功率的关系
# plt.figure(figsize=(12, 6))
# plt.scatter(train_with_predictions['power'], train_with_predictions['abs_residual'], alpha=0.5)
# plt.xlabel('功率')
# plt.ylabel('绝对残差')
# plt.title('功率与预测残差的关系')
# plt.savefig('residual_vs_power.png')
# print('残差与功率的关系图已保存为 residual_vs_power.png')

# # 残差与里程的关系
# plt.figure(figsize=(12, 6))
# plt.scatter(train_with_predictions['km'], train_with_predictions['abs_residual'], alpha=0.5)
# plt.xlabel('里程')
# plt.ylabel('绝对残差')
# plt.title('里程与预测残差的关系')
# plt.savefig('residual_vs_km.png')
# print('残差与里程的关系图已保存为 residual_vs_km.png')

# # 残差与车龄的关系
# plt.figure(figsize=(12, 6))
# plt.scatter(train_with_predictions['car_age_days'], train_with_predictions['abs_residual'], alpha=0.5)
# plt.xlabel('车龄（天）')
# plt.ylabel('绝对残差')
# plt.title('车龄与预测残差的关系')
# plt.savefig('residual_vs_car_age.png')
# print('残差与车龄的关系图已保存为 residual_vs_car_age.png')

# # 查看误差最大的样本
# top_error_samples = train_with_predictions.sort_values('abs_residual', ascending=False).head(20)
# print("\n误差最大的样本：")
# print(top_error_samples[['price', 'pred_price', 'residual', 'abs_residual', 'power', 'km', 'car_age_days']])

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
# 获取测试数据ID
if test_ids is not None:
    print(f'从外部获取测试数据ID，数量: {len(test_ids)}')
    out_df['ID'] = test_ids
elif test is not None and 'ID' in test.columns:
    print(f'从测试数据中获取ID，数量: {len(test)}')
    out_df['ID'] = test['ID']
else:
    # 直接从原test文件读取ID
    import os
    import pandas as pd
    test_files = ['data/testA.csv']
    test_id_loaded = False
    for test_file in test_files:
        test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_file)
        if os.path.exists(test_file_path):
            try:
                test_data = pd.read_csv(test_file_path)
                if 'ID' in test_data.columns:
                    out_df['ID'] = test_data['ID']
                    print(f'从{test_file}读取测试数据ID')
                    test_id_loaded = True
                    break
            except Exception as e:
                print(f'读取{test_file}时出错: {e}')
    
    if not test_id_loaded:
        # 如果还是无法读取，生成默认ID
        out_df['ID'] = range(len(y_pred_test_stacking))
        print('注意：无法获取原始测试数据ID，使用默认ID')
out_df['price'] = best_y_pred_test

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
