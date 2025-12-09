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
    original_df = df.copy()  # 保留原始数据用于特征工程

    # 删除duration特征（会造成数据泄露）
    if 'duration' in data.columns:
        data = data.drop('duration', axis=1)

    # 处理分类变量 - 先保存原始分类信息用于特征工程
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

    # 高级特征工程
    # 1. 账户余额的高级特征
    data['balance_bin'] = pd.qcut(data['balance'], q=4, labels=False, duplicates='drop')
    data['balance_log'] = np.log1p(data['balance'])  # 对数变换，log1p确保安全性
    data['is_high_balance'] = (data['balance'] > data['balance'].quantile(0.75)).astype(int)  # 高余额标记
    
    # 2. pdays的高级处理
    data['pdays'] = np.where(data['pdays'] == -1, 0, data['pdays'])  # 处理-1值
    data['pdays_sqrt'] = np.sqrt(data['pdays'] + 1)  # 平方根变换
    data['pdays_reciprocal'] = 1 / (data['pdays'] + 1)  # 倒数变换，+1避免除以0
    data['is_recent_contact'] = (data['pdays'] <= 30).astype(int)  # 最近联系标记
    
    # 3. 联系次数的衍生特征
    data['total_contacts'] = data['campaign'] + data['previous']
    data['is_first_contact'] = (data['previous'] == 0).astype(int)  # 是否首次联系
    data['contact_frequency'] = np.where(data['pdays'] == 0, data['campaign'], data['campaign'] / (data['pdays'] + 1))
    data['is_high_contact'] = (data['campaign'] > 3).astype(int)  # 高联系频率标记
    
    # 4. 年龄的高级特征
    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=False, right=False)  # 更细的年龄分组
    data['is_young_adult'] = ((data['age'] >= 25) & (data['age'] <= 40)).astype(int)  # 年轻成年人
    data['is_senior'] = (data['age'] >= 60).astype(int)  # 老年人
    
    # 5. 贷款状况的组合特征
    if 'housing' in original_df.columns and 'loan' in original_df.columns:
        data['loan_status'] = original_df['housing'].astype(str) + '_' + original_df['loan'].astype(str)
        data = pd.get_dummies(data, columns=['loan_status'], drop_first=True)
    
    # 6. 教育和婚姻状况的组合特征
    if 'education' in original_df.columns and 'marital' in original_df.columns:
        data['edu_marital'] = original_df['education'].astype(str) + '_' + original_df['marital'].astype(str)
        data = pd.get_dummies(data, columns=['edu_marital'], drop_first=True)
    
    # 7. 月份的高级特征
    if 'month' in original_df.columns:
        month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                      'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        data['month_order'] = original_df['month'].map(month_order)
        data['is_quarter_end'] = data['month_order'].isin([3, 6, 9, 12]).astype(int)
        data['is_holiday_season'] = data['month_order'].isin([11, 12, 1]).astype(int)
    
    # 8. 职业和教育的组合特征
    if 'job' in original_df.columns and 'education' in original_df.columns:
        data['job_edu'] = original_df['job'].astype(str) + '_' + original_df['education'].astype(str)
        data = pd.get_dummies(data, columns=['job_edu'], drop_first=True)
    
    # 9. 统计特征 - 添加安全检查
    data['balance_per_age'] = data['balance'] / (data['age'] + 1)  # +1避免除以0
    data['contacts_per_age'] = data['total_contacts'] / (data['age'] + 1)  # +1避免除以0
    
    # 10. 交互特征 - 添加特征裁剪
    data['balance_pdays_interaction'] = data['balance'] * data['pdays']
    data['age_campaign_interaction'] = data['age'] * data['campaign']
    data['balance_campaign_interaction'] = data['balance'] * data['campaign']
    
    # 11. 特征裁剪 - 确保所有数值特征都是有限的
    for col in data.select_dtypes(include=[np.number]).columns:
        # 替换无穷大值为NaN
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        # 用列的中位数填充NaN
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())
        # 裁剪极端值到合理范围（1%到99%分位数）
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

# 测试集特征
X_test = test_processed

# 更新数值特征列表，包含所有新添加的数值特征
numeric_cols = [
    'age', 'balance', 'day', 'campaign', 'pdays', 'previous', 
    'total_contacts', 'is_first_contact', 'age_group', 'balance_bin',
    'balance_log', 'is_high_balance', 'pdays_sqrt', 'pdays_reciprocal', 
    'is_recent_contact', 'contact_frequency', 'is_high_contact', 
    'is_young_adult', 'is_senior', 'balance_per_age', 'contacts_per_age',
    'balance_pdays_interaction', 'age_campaign_interaction', 'balance_campaign_interaction'
]

# 添加月份相关特征
if 'month_order' in X_train.columns:
    numeric_cols.extend(['month_order', 'is_quarter_end', 'is_holiday_season'])

# 标准化处理数值特征
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 优化1：特征选择 - 使用VarianceThreshold去除低方差特征
print("\n进行特征选择...")
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

# 去除低方差特征
var_threshold = VarianceThreshold(threshold=0.01)
X_train_var = var_threshold.fit_transform(X_train)
X_test_var = var_threshold.transform(X_test)
print(f"原始特征数量: {X_train.shape[1]}, 去除低方差特征后: {X_train_var.shape[1]}")

# 转换回DataFrame，保持列名
selected_cols = X_train.columns[var_threshold.get_support()]
X_train_selected = pd.DataFrame(X_train_var, columns=selected_cols, index=X_train.index)
X_test_selected = pd.DataFrame(X_test_var, columns=selected_cols, index=X_test.index)

# 处理数据不平衡问题 - 使用更高级的过采样策略
print("\n使用SMOTE处理数据不平衡...")
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN

# 使用ADASYN替代SMOTE，生成更自然的合成样本
adasyn = ADASYN(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_selected, y_train)
print(f"原始数据分布: {dict(pd.Series(y_train).value_counts())}")
print(f"过采样后数据分布: {dict(pd.Series(y_train_resampled).value_counts())}")

# 优化1：使用更复杂的CatBoost参数（修复参数冲突）
print("\n使用优化的CatBoost模型进行训练...")
best_model = CatBoostClassifier(
    random_state=42, 
    iterations=400, 
    learning_rate=0.05, 
    depth=8, 
    auto_class_weights='Balanced',
    subsample=0.85,
    colsample_bylevel=0.85,
    l2_leaf_reg=5,  # 只保留l2_leaf_reg，移除reg_lambda（同义词冲突）
    bootstrap_type='Bernoulli',
    grow_policy='SymmetricTree',
    verbose=0
)

# 训练模型
best_model.fit(X_train_resampled, y_train_resampled)

# 评估模型
final_train_score = accuracy_score(y_train, best_model.predict(X_train_selected))
print(f"CatBoost模型在训练集上的准确率: {final_train_score:.4f}")

# 优化2：使用更高级的Stacking Ensemble
print("\n尝试使用高级Stacking Ensemble...")
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression as StackingMetaModel

# 准备更强的基础模型
base_estimators = [
    ('rf', RandomForestClassifier(
        random_state=42, 
        n_estimators=300, 
        max_depth=12, 
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        bootstrap=True
    )),
    ('gb', GradientBoostingClassifier(
        random_state=42, 
        n_estimators=400, 
        learning_rate=0.05, 
        max_depth=7,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.85,
        max_features='sqrt',
        verbose=0
    )),
    ('catboost', best_model)  # 使用已经训练好的CatBoost模型
]

# 使用更复杂的元分类器
meta_model = LogisticRegression(
    random_state=42, 
    max_iter=3000, 
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    class_weight='balanced'
)

# 创建Stacking模型
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=True  # 传递原始特征给元分类器
)

# 训练Stacking模型
stacking_model.fit(X_train_resampled, y_train_resampled)

# 评估Stacking模型
stacking_train_score = accuracy_score(y_train, stacking_model.predict(X_train_selected))
print(f"Stacking Ensemble在训练集上的准确率: {stacking_train_score:.4f}")

# 优化3：使用Voting Ensemble作为备选
print("\n尝试使用Voting Ensemble...")
from sklearn.ensemble import VotingClassifier

voting_model = VotingClassifier(
    estimators=[
        ('catboost', best_model),
        ('stacking', stacking_model)
    ],
    voting='soft',
    n_jobs=-1
)

voting_model.fit(X_train_resampled, y_train_resampled)

# 评估Voting模型
voting_train_score = accuracy_score(y_train, voting_model.predict(X_train_selected))
print(f"Voting Ensemble在训练集上的准确率: {voting_train_score:.4f}")

# 选择表现最好的模型
scores = {
    'CatBoost': final_train_score,
    'Stacking': stacking_train_score,
    'Voting': voting_train_score
}

best_ensemble_name = max(scores, key=scores.get)
if best_ensemble_name == 'CatBoost':
    final_model = best_model
elif best_ensemble_name == 'Stacking':
    final_model = stacking_model
else:
    final_model = voting_model

print(f"\n选择{best_ensemble_name} Ensemble作为最终模型")

# 预测
y_pred = final_model.predict(X_test_selected)

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
