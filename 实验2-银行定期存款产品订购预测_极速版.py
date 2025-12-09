#!/usr/bin/env python
# coding: utf-8

# # 实验2-银行定期存款产品订购预测 - 极速版

# 输入学号
student_id= '25451354008'

# In[1]:


# === 1. 导入必要的库 ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import datetime
import warnings
warnings.filterwarnings('ignore')

# 引入高效模型
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 引入不平衡数据处理库
from imblearn.over_sampling import ADASYN

# 引入特征选择和随机森林
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

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


# ## 数据预处理

def preprocess_data(df):
    data = df.copy()
    original_df = df.copy()

    # 删除duration特征（会造成数据泄露）
    if 'duration' in data.columns:
        data = data.drop('duration', axis=1)

    # 处理分类变量 - 独热编码
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'month']

    # 处理未知值
    for col in categorical_columns:
        if col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].replace('unknown', np.nan)
                if data[col].isna().any():
                    mode_val = data[col].mode()
                    if len(mode_val) > 0:
                        data[col] = data[col].fillna(mode_val[0])

    # 独热编码
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # 高级特征工程
    # 1. 账户余额特征
    data['balance_bin'] = pd.qcut(data['balance'], q=4, labels=False, duplicates='drop')
    data['balance_log'] = np.log1p(data['balance'])
    data['is_high_balance'] = (data['balance'] > data['balance'].quantile(0.75)).astype(int)
    
    # 2. pdays处理
    data['pdays'] = np.where(data['pdays'] == -1, 0, data['pdays'])
    data['pdays_sqrt'] = np.sqrt(data['pdays'] + 1)
    data['is_recent_contact'] = (data['pdays'] <= 30).astype(int)
    
    # 3. 联系次数特征
    data['total_contacts'] = data['campaign'] + data['previous']
    data['is_first_contact'] = (data['previous'] == 0).astype(int)
    data['contact_frequency'] = np.where(data['pdays'] == 0, data['campaign'], data['campaign'] / (data['pdays'] + 1))
    
    # 4. 年龄特征
    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=False, right=False)
    
    # 5. 月份特征
    if 'month' in original_df.columns:
        month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                      'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        data['month_order'] = original_df['month'].map(month_order)
    
    # 6. 统计特征
    data['balance_per_age'] = data['balance'] / (data['age'] + 1)
    data['contacts_per_age'] = data['total_contacts'] / (data['age'] + 1)
    
    # 7. 特征安全处理
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())
        lower = data[col].quantile(0.01)
        upper = data[col].quantile(0.99)
        data[col] = data[col].clip(lower, upper)

    return data

# 预处理训练集和测试集
train_processed = preprocess_data(train)
test_processed = preprocess_data(test)

# 准备特征和目标变量
X_train = train_processed.drop('deposit', axis=1)
y_train = train_processed['deposit'].map({'yes': 1, 'no': 0})
X_test = test_processed

# 数值特征标准化
numeric_cols = [col for col in X_train.columns if X_train[col].dtype in [np.int64, np.float64]]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 特征选择 - RFE选择30个最重要特征
print("\n进行RFE特征选择...")
selector = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    n_features_to_select=30,
    step=5,
    verbose=1
)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_cols = X_train.columns[selector.support_]
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_cols, index=X_train.index)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_cols, index=X_test.index)
print(f"特征选择完成: {X_train.shape[1]} → {X_train_selected_df.shape[1]}")

# 处理数据不平衡 - ADASYN
print("\n使用ADASYN处理数据不平衡...")
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_selected_df, y_train)
print(f"过采样后数据分布: {dict(pd.Series(y_train_resampled).value_counts())}")

# 训练顶级模型 - CatBoost
print("\n训练CatBoost模型...")
catboost_model = CatBoostClassifier(
    random_state=42,
    iterations=400,
    learning_rate=0.05,
    depth=8,
    auto_class_weights='Balanced',
    subsample=0.85,
    colsample_bylevel=0.85,
    l2_leaf_reg=3,
    bootstrap_type='Bernoulli',
    grow_policy='SymmetricTree',
    verbose=0
)
catboost_model.fit(X_train_resampled, y_train_resampled)
catboost_score = accuracy_score(y_train, catboost_model.predict(X_train_selected_df))
print(f"CatBoost准确率: {catboost_score:.4f}")

# 训练顶级模型 - LightGBM
print("\n训练LightGBM模型...")
lgbm_model = LGBMClassifier(
    random_state=42,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=64,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=3,
    class_weight='balanced',
    verbose=-1
)
lgbm_model.fit(X_train_resampled, y_train_resampled)
lgbm_score = accuracy_score(y_train, lgbm_model.predict(X_train_selected_df))
print(f"LightGBM准确率: {lgbm_score:.4f}")

# 使用Voting Ensemble组合最佳模型
print("\n创建Voting Ensemble...")
voting_model = VotingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft',
    n_jobs=-1
)
voting_model.fit(X_train_resampled, y_train_resampled)
voting_score = accuracy_score(y_train, voting_model.predict(X_train_selected_df))
print(f"Voting Ensemble准确率: {voting_score:.4f}")

# 选择最佳模型
models = {
    'CatBoost': (catboost_model, catboost_score),
    'LightGBM': (lgbm_model, lgbm_score),
    'Voting Ensemble': (voting_model, voting_score)
}

best_model_name = max(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]
best_score = models[best_model_name][1]
print(f"\n最佳模型: {best_model_name}, 准确率: {best_score:.4f}")

# 预测
y_pred = best_model.predict(X_test_selected_df)

# 保存预测结果
print("\n生成预测结果...")
submission = pd.DataFrame({'prediction': y_pred})
submission['prediction'] = submission['prediction'].map({1: 'yes', 0: 'no'})
subdir = ''
submission.to_csv(subdir + student_id + 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False)

print("\n模型训练和预测完成！")
print(f"最终模型: {best_model_name}")
print(f"最终准确率: {best_score:.4f}")
