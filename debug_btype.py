import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 读取数据
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/testA.csv')

# 模拟时间特征提取的日期处理
def process_date(date_str, format='%Y%m%d'):
    try:
        if len(date_str) != 8:
            return pd.NaT
        if date_str[4:6] == '00' or date_str[6:8] == '00':
            new_date_str = date_str[:4] + '01' + date_str[6:8] if date_str[4:6] == '00' else date_str[:6] + '01'
            return pd.to_datetime(new_date_str, format=format)
        elif date_str == '20220229':
            return pd.to_datetime('20220228', format=format)
        else:
            return pd.to_datetime(date_str, format=format)
    except:
        return pd.NaT

# 转换日期
train['regDate'] = train['regDate'].astype(str).apply(process_date)
train['creatDate'] = train['creatDate'].astype(str).apply(process_date)
test['regDate'] = test['regDate'].astype(str).apply(process_date)
test['creatDate'] = test['creatDate'].astype(str).apply(process_date)

# 提取时间特征
train['reg_year'] = train['regDate'].dt.year
train['reg_month'] = train['regDate'].dt.month
train['reg_quarter'] = train['regDate'].dt.quarter
train['creat_year'] = train['creatDate'].dt.year
train['creat_month'] = train['creatDate'].dt.month
train['creat_quarter'] = train['creatDate'].dt.quarter
train['creat_dayofweek'] = train['creatDate'].dt.dayofweek
train['car_age_days'] = (train['creatDate'] - train['regDate']).dt.days

test['reg_year'] = test['regDate'].dt.year
test['reg_month'] = test['regDate'].dt.month
test['reg_quarter'] = test['regDate'].dt.quarter
test['creat_year'] = test['creatDate'].dt.year
test['creat_month'] = test['creatDate'].dt.month
test['creat_quarter'] = test['creatDate'].dt.quarter
test['creat_dayofweek'] = test['creatDate'].dt.dayofweek
test['car_age_days'] = (test['creatDate'] - test['regDate']).dt.days

# 选择特征列
numerical_cols = train.select_dtypes(exclude = 'object').columns
feature_cols = [
    col for col in numerical_cols if col not in [
        'ID', 'name', 'regDate', 'creatDate', 'price', 'model', 'brand',
        'regionCode', 'seller'
    ]
]
feature_cols.extend(['BType', 'FType'])

# 异常值处理
power_upper = train['power'].quantile(0.999)
train.loc[train['power'] > power_upper, 'power'] = power_upper
test.loc[test['power'] > power_upper, 'power'] = power_upper

# 查看BType和FType在处理前的情况
print('=== 处理前 ===')
print(f'BType唯一值数量: {train["BType"].nunique()}')
print(f'BType唯一值: {train["BType"].unique()}')
print(f'FType唯一值数量: {train["FType"].nunique()}')
print(f'FType唯一值: {train["FType"].unique()}')

# 准备数据
X_train = train[feature_cols]
y_train = train['price']
X_test = test[feature_cols]

# 缺失值处理
print('\n=== 缺失值处理 ===')
categorical_cols = ['gearbox', 'BType', 'FType']
continuous_cols = [col for col in feature_cols if col not in categorical_cols]

# 离散型特征使用均值填充
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='mean')
    X_train_cat = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test_cat = cat_imputer.transform(X_test[categorical_cols])
    print(f'离散型特征 {categorical_cols} 使用均值填充，均值为: {cat_imputer.statistics_}')

# 连续型特征使用中位数填充
if continuous_cols:
    cont_imputer = SimpleImputer(strategy='median')
    X_train_cont = cont_imputer.fit_transform(X_train[continuous_cols])
    X_test_cont = cont_imputer.transform(X_test[continuous_cols])

# 合并特征
if categorical_cols and continuous_cols:
    X_train = np.hstack((X_train_cont, X_train_cat))
    X_test = np.hstack((X_test_cont, X_test_cat))

# 查看填充后的BType和FType
print('\n=== 填充后 ===')
# 获取BType和FType在X_train中的位置
btype_idx = feature_cols.index('BType')
ftype_idx = feature_cols.index('FType')
# 计算它们在合并后的X_train中的位置（连续特征数 + 它们在categorical_cols中的位置）
cat_start_idx = len(continuous_cols)
btype_pos = cat_start_idx + categorical_cols.index('BType')
ftype_pos = cat_start_idx + categorical_cols.index('FType')

print(f'BType在X_train中的位置: {btype_pos}')
print(f'FType在X_train中的位置: {ftype_pos}')

# 查看填充后的BType值
filled_btype = X_train[:, btype_pos]
filled_ftype = X_train[:, ftype_pos]

print(f'填充后BType唯一值数量: {len(np.unique(filled_btype))}')
print(f'填充后BType前10个值: {filled_btype[:10]}')
print(f'填充后FType唯一值数量: {len(np.unique(filled_ftype))}')
print(f'填充后FType前10个值: {filled_ftype[:10]}')

# 检查是否存在非整数
print(f'\nBType中非整数值数量: {np.sum(filled_btype != np.round(filled_btype))}')
print(f'FType中非整数值数量: {np.sum(filled_ftype != np.round(filled_ftype))}')