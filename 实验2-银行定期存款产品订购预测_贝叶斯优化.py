#!/usr/bin/env python
# coding: utf-8

# # 实验2-银行定期存款产品订购预测 - 贝叶斯优化版

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

# 引入特征选择
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# 引入贝叶斯优化
from bayes_opt import BayesianOptimization

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

# 贝叶斯优化LightGBM参数
def lgb_eval(n_estimators, learning_rate, max_depth, num_leaves, subsample, colsample_bytree, reg_lambda):
    model = LGBMClassifier(
        random_state=42,
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        num_leaves=int(num_leaves),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        class_weight='balanced',
        verbose=-1
    )
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_train_selected_df)
    return accuracy_score(y_train, y_pred)

# 定义参数搜索空间
lgb_params = {
    'n_estimators': (300, 600),
    'learning_rate': (0.01, 0.1),
    'max_depth': (6, 10),
    'num_leaves': (32, 128),
    'subsample': (0.7, 0.95),
    'colsample_bytree': (0.7, 0.95),
    'reg_lambda': (0.1, 10)
}

# 执行贝叶斯优化
print("\n开始贝叶斯优化LightGBM参数...")
lgb_bo = BayesianOptimization(
    f=lgb_eval,
    pbounds=lgb_params,
    random_state=42,
    verbose=2
)

# 运行优化
lgb_bo.maximize(init_points=5, n_iter=20)

# 获取最佳参数
best_params = lgb_bo.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['num_leaves'] = int(best_params['num_leaves'])

print(f"\n最佳参数: {best_params}")
print(f"最佳准确率: {lgb_bo.max['target']:.4f}")

# 使用最佳参数训练LightGBM模型
print("\n使用最佳参数训练LightGBM模型...")
best_lgb_model = LGBMClassifier(
    random_state=42,
    class_weight='balanced',
    verbose=-1,
    **best_params
)
best_lgb_model.fit(X_train_resampled, y_train_resampled)

# 评估模型
best_lgb_score = accuracy_score(y_train, best_lgb_model.predict(X_train_selected_df))
print(f"最佳LightGBM模型准确率: {best_lgb_score:.4f}")

# 训练CatBoost模型进行比较
print("\n训练CatBoost模型进行比较...")
best_cat_model = CatBoostClassifier(
    random_state=42,
    iterations=500,
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
best_cat_model.fit(X_train_resampled, y_train_resampled)
best_cat_score = accuracy_score(y_train, best_cat_model.predict(X_train_selected_df))
print(f"CatBoost模型准确率: {best_cat_score:.4f}")

# 训练Voting Ensemble
print("\n训练Voting Ensemble...")
voting_model = VotingClassifier(
    estimators=[
        ('lgb', best_lgb_model),
        ('cat', best_cat_model)
    ],
    voting='soft',
    n_jobs=-1
)
voting_model.fit(X_train_resampled, y_train_resampled)
voting_score = accuracy_score(y_train, voting_model.predict(X_train_selected_df))
print(f"Voting Ensemble准确率: {voting_score:.4f}")

# 选择最终模型
final_models = {
    'LightGBM': (best_lgb_model, best_lgb_score),
    'CatBoost': (best_cat_model, best_cat_score),
    'Voting Ensemble': (voting_model, voting_score)
}

final_best_name = max(final_models, key=lambda x: final_models[x][1])
final_best_model = final_models[final_best_name][0]
final_best_score = final_models[final_best_name][1]

print(f"\n最终最佳模型: {final_best_name}")
print(f"最终准确率: {final_best_score:.4f}")

# 预测
y_pred = final_best_model.predict(X_test_selected_df)

# 保存预测结果
print("\n生成预测结果...")
submission = pd.DataFrame({'prediction': y_pred})
submission['prediction'] = submission['prediction'].map({1: 'yes', 0: 'no'})
subdir = ''
submission.to_csv(subdir + student_id + 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False)

print("\n模型训练和预测完成！")
print(f"最终模型: {final_best_name}")
print(f"最终准确率: {final_best_score:.4f}")
