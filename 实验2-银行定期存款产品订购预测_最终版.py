#!/usr/bin/env python
# coding: utf-8

# # 实验2-银行定期存款产品订购预测

# ## 学号：
# ## 姓名：
# ## 班级：

# 输入学号
student_id= '25451354008'

# In[1]:


# === 1. 导入必要的库 ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import datetime
import warnings
warnings.filterwarnings('ignore')

# 引入先进模型
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 引入不平衡数据处理库
from imblearn.over_sampling import SMOTE

# ## 读取数据

# In[2]:


# path to where the data lies
dpath = 'data/'
train=pd.read_csv(dpath +"train.csv")
test=pd.read_csv(dpath +"testA.csv")


# In[3]:


## 输出数据的大小信息
print('Train data shape:',train.shape)
print('TestA data shape:',test.shape)
print("\n目标变量 'deposit' 的分布:")
print(train['deposit'].value_counts())


# ## 数据建模部分

# In[4]:


# 数据预处理
def preprocess_data(df):
    data = df.copy()

    # 删除duration特征（会造成数据泄露）
    if 'duration' in data.columns:
        data = data.drop('duration', axis=1)

    # 处理分类变量 - 使用独热编码
    categorical_columns = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'poutcome', 'month'
    ]

    # 处理未知值
    for col in categorical_columns:
        if col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].replace('unknown', np.nan)
                # 用众数填充缺失值
                if data[col].isna().any():
                    mode_val = data[col].mode()
                    if len(mode_val) > 0:
                        data[col] = data[col].fillna(mode_val[0])

    # 使用独热编码处理分类变量
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # 添加衍生特征
    # 1. balance的分箱特征
    data['balance_bin'] = pd.qcut(data['balance'], q=4, labels=False, duplicates='drop')

    # 2. 上次联系到本次联系的时间间隔的分类
    data['pdays'] = np.where(data['pdays'] == -1, 0, data['pdays'])  # 处理-1值

    # 3. 联系次数的衍生特征
    data['total_contacts'] = data['campaign'] + data['previous']
    data['is_first_contact'] = (data['previous'] == 0).astype(int)  # 是否首次联系

    # 4. 年龄的分箱特征
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 100], labels=False, right=False)

    return data

# 预处理训练集和测试集
train_processed = preprocess_data(train)
test_processed = preprocess_data(test)

# 准备特征和目标变量
X_train = train_processed.drop('deposit', axis=1)
y_train = train_processed['deposit'].map({'yes': 1, 'no': 0})

# 测试集特征
X_test = test_processed

# 数值特征标准化
numeric_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous', 
                'total_contacts', 'is_first_contact', 'age_group', 'balance_bin']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 处理数据不平衡问题 - 使用SMOTE
print("\n使用SMOTE处理数据不平衡...")
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"原始数据分布: {dict(pd.Series(y_train).value_counts())}")
print(f"过采样后数据分布: {dict(pd.Series(y_train_resampled).value_counts())}")

# 优化1：使用更复杂的CatBoost参数
print("\n使用优化的CatBoost模型进行训练...")
best_model = CatBoostClassifier(
    random_state=42, 
    iterations=300, 
    learning_rate=0.08, 
    depth=7, 
    auto_class_weights='Balanced',
    subsample=0.8,
    colsample_bylevel=0.8,
    reg_lambda=3,
    bootstrap_type='Bernoulli',
    verbose=0
)

# 训练模型
best_model.fit(X_train_resampled, y_train_resampled)

# 评估模型
final_train_score = accuracy_score(y_train, best_model.predict(X_train))
print(f"CatBoost模型在训练集上的准确率: {final_train_score:.4f}")

# 优化2：使用Stacking Ensemble进一步提升性能
print("\n尝试使用Stacking Ensemble...")
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression as StackingMetaModel

# 准备基础模型
base_estimators = [
    ('rf', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, class_weight='balanced')),
    ('gb', GradientBoostingClassifier(random_state=42, n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8)),
    ('catboost', best_model)  # 使用已经训练好的CatBoost模型
]

# 创建Stacking模型
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=StackingMetaModel(random_state=42, max_iter=2000),
    cv=5,
    n_jobs=-1
)

# 训练Stacking模型
stacking_model.fit(X_train_resampled, y_train_resampled)

# 评估Stacking模型
stacking_train_score = accuracy_score(y_train, stacking_model.predict(X_train))
print(f"Stacking Ensemble在训练集上的准确率: {stacking_train_score:.4f}")

# 选择表现更好的模型
if stacking_train_score > final_train_score:
    final_model = stacking_model
    print("\n选择Stacking Ensemble作为最终模型")
else:
    final_model = best_model
    print("\n选择CatBoost作为最终模型")

# 预测
y_pred = final_model.predict(X_test)

print("\n模型训练完成！")


# ## 提交结果文件
# 这里不用改动，自动会记录提交的时间

# In[5]:


# 保存预测结果
submission = pd.DataFrame({'prediction': y_pred})
submission['prediction'] = submission['prediction'].map({1: 'yes', 0: 'no'})
subdir = ''
submission.to_csv(subdir + student_id + 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False)

print("预测结果已生成！")

# 最后把csv文件上传到ftp://10.132.219.5:955 的相应目录
