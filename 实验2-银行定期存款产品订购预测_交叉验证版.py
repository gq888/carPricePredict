#!/usr/bin/env python
# coding: utf-8

# # 实验2-银行定期存款产品订购预测 - 交叉验证版
# ## 解决过拟合问题，只显示交叉验证结果

# 输入学号
student_id= '25451354008'

# === 1. 导入必要的库 ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import datetime
import warnings
warnings.filterwarnings('ignore')

# 引入先进模型
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 引入不平衡数据处理库
from imblearn.over_sampling import ADASYN

# 引入特征选择
from sklearn.feature_selection import VarianceThreshold, RFE

# 确保所有模型都已正确导入
print("已导入的模型: LightGBM, XGBoost, CatBoost, RandomForest, GradientBoosting, SVM")

# ## 读取数据
# path to where the data lies
dpath = 'data/'
train=pd.read_csv(dpath +"train.csv")
test=pd.read_csv(dpath +"testA.csv")

## 输出数据的大小信息
print('\nTrain data shape:', train.shape)
print('TestA data shape:', test.shape)
print("\n目标变量 'deposit' 的分布:")
print(train['deposit'].value_counts())

# ## 数据预处理
def preprocess_data(df):
    data = df.copy()
    original_df = df.copy()

    # 删除duration特征（会造成数据泄露）
    if 'duration' in data.columns:
        data = data.drop('duration', axis=1)

    # 处理分类变量 - 先保存原始分类信息用于高级特征工程
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
    
    # 1. 账户余额的高级特征
    data['balance_log'] = np.log1p(data['balance'])  # 对数变换，处理长尾分布
    data['balance_bin'] = pd.qcut(data['balance'], q=5, labels=False, duplicates='drop')  # 更细的分箱
    data['is_high_balance'] = (data['balance'] > data['balance'].quantile(0.8)).astype(int)  # 高余额标记
    data['is_low_balance'] = (data['balance'] <= 0).astype(int)  # 低余额标记
    
    # 2. 年龄的高级特征
    data['age_group'] = pd.cut(data['age'], bins=[0, 20, 30, 40, 50, 60, 70, 100], labels=False, right=False)  # 更细的年龄分组
    data['is_young_adult'] = ((data['age'] >= 25) & (data['age'] <= 40)).astype(int)  # 年轻成年人
    data['is_senior'] = (data['age'] >= 65).astype(int)  # 老年人
    data['is_retired_age'] = (data['age'] >= 60).astype(int)  # 退休年龄
    
    # 3. 联系相关的高级特征
    data['total_contacts'] = data['campaign'] + data['previous']
    data['is_first_contact'] = (data['previous'] == 0).astype(int)
    data['is_frequent_contact'] = (data['campaign'] > 3).astype(int)  # 频繁联系标记
    data['contact_ratio'] = data['previous'] / (data['campaign'] + 1)  # 历史联系成功率比例
    
    # 4. pdays的高级处理
    data['pdays'] = np.where(data['pdays'] == -1, 0, data['pdays'])
    data['is_recent_contact'] = (data['pdays'] <= 30).astype(int)
    data['is_long_ago_contact'] = (data['pdays'] > 90).astype(int)  # 长时间未联系
    data['pdays_sqrt'] = np.sqrt(data['pdays'] + 1)  # 平方根变换，处理长尾分布
    data['pdays_reciprocal'] = 1 / (data['pdays'] + 1)  # 倒数变换，突出近期联系
    
    # 5. 月份和日期的高级特征
    if 'month' in original_df.columns:
        # 月份数值映射
        month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                      'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        original_df['month_num'] = original_df['month'].map(month_order)
        
        # 季度特征
        data['quarter'] = (original_df['month_num'] - 1) // 3 + 1
        data['is_quarter_end'] = original_df['month_num'].isin([3, 6, 9, 12]).astype(int)
        
        # 季节特征
        data['season'] = original_df['month_num'].apply(lambda x: 1 if x in [12,1,2] else 2 if x in [3,4,5] else 3 if x in [6,7,8] else 4)
        
        # 节假日季节
        data['is_holiday_season'] = original_df['month_num'].isin([11, 12, 1]).astype(int)
    
    # 6. 贷款状况的组合特征
    if 'housing' in original_df.columns and 'loan' in original_df.columns:
        data['loan_status'] = original_df['housing'].astype(str) + '_' + original_df['loan'].astype(str)
        data = pd.get_dummies(data, columns=['loan_status'], drop_first=True)
    
    # 7. 教育和婚姻状况的组合特征
    if 'education' in original_df.columns and 'marital' in original_df.columns:
        data['edu_marital'] = original_df['education'].astype(str) + '_' + original_df['marital'].astype(str)
        data = pd.get_dummies(data, columns=['edu_marital'], drop_first=True)
    
    # 8. 职业和教育的组合特征
    if 'job' in original_df.columns and 'education' in original_df.columns:
        data['job_edu'] = original_df['job'].astype(str) + '_' + original_df['education'].astype(str)
        data = pd.get_dummies(data, columns=['job_edu'], drop_first=True)
    
    # 9. 交互特征
    data['age_balance_interaction'] = data['age'] * data['balance_log']  # 年龄和余额的交互
    data['age_contacts_interaction'] = data['age'] * data['total_contacts']  # 年龄和联系次数的交互
    data['balance_contacts_interaction'] = data['balance_log'] * data['total_contacts']  # 余额和联系次数的交互
    
    # 10. 统计特征
    data['balance_per_age'] = data['balance'] / (data['age'] + 1)  # 年龄归一化的余额
    data['contacts_per_age'] = data['total_contacts'] / (data['age'] + 1)  # 年龄归一化的联系次数
    
    # 11. 二分类特征的组合
    if 'default' in original_df.columns and 'housing' in original_df.columns:
        # 注意：这里需要重新获取原始的二分类特征值，因为已经被独热编码了
        # 我们需要从原始数据中提取这些特征
        original_df['default_bin'] = (original_df['default'] == 'yes').astype(int)
        original_df['housing_bin'] = (original_df['housing'] == 'yes').astype(int)
        original_df['loan_bin'] = (original_df['loan'] == 'yes').astype(int)
        
        data['risk_profile'] = original_df['default_bin'] + original_df['housing_bin'] + original_df['loan_bin']  # 风险评分
    
    # 12. 特征安全处理
    for col in data.select_dtypes(include=[np.number]).columns:
        # 替换无穷大值为NaN
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        # 用中位数填充NaN
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())
        # 裁剪极端值到1%-99%分位数
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

# 特征选择 - 组合方差过滤和RFE
print("\n进行特征选择...")

# 步骤1：去除低方差特征
var_threshold = VarianceThreshold(threshold=0.01)
X_train_var = var_threshold.fit_transform(X_train)
X_test_var = var_threshold.transform(X_test)
var_selected_cols = X_train.columns[var_threshold.get_support()]
X_train_var_df = pd.DataFrame(X_train_var, columns=var_selected_cols, index=X_train.index)
X_test_var_df = pd.DataFrame(X_test_var, columns=var_selected_cols, index=X_test.index)
print(f"步骤1 - 去除低方差特征: {X_train.shape[1]} → {X_train_var_df.shape[1]}")

# 步骤2：使用RFE进行递归特征消除
print("步骤2 - 使用RFE进行递归特征消除...")
from sklearn.ensemble import RandomForestClassifier

# 使用RandomForest作为基础评估器
estimator = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1, verbose=0)

# RFE选择最重要的特征，保留约30个
n_features = min(30, X_train_var_df.shape[1] // 2)
print(f"RFE目标特征数量: {n_features}")

rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=5, verbose=0)
rfe.fit(X_train_var_df, y_train)

X_train_selected = rfe.transform(X_train_var_df)
X_test_selected = rfe.transform(X_test_var_df)

# 转换回DataFrame，保持列名
rfe_selected_cols = var_selected_cols[rfe.support_]
X_train_selected_df = pd.DataFrame(X_train_selected, columns=rfe_selected_cols, index=X_train.index)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=rfe_selected_cols, index=X_test.index)
print(f"步骤2 - RFE特征选择: {X_train_var_df.shape[1]} → {X_train_selected_df.shape[1]}")
print(f"最终特征选择完成: {X_train.shape[1]} → {X_train_selected_df.shape[1]}")

# 处理数据不平衡 - ADASYN
print("\n使用ADASYN处理数据不平衡...")
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_selected_df, y_train)
print(f"过采样后数据分布: {dict(pd.Series(y_train_resampled).value_counts())}")

# 设置交叉验证策略 - 5折分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n使用5折分层交叉验证评估所有模型...")

# 定义模型列表
models = {
    'RandomForest': RandomForestClassifier(
        random_state=42,
        n_estimators=500,  # 增加树的数量
        max_depth=15,  # 增加树的深度
        min_samples_split=4,  # 调整分裂样本数
        min_samples_leaf=2,  # 调整叶子节点样本数
        class_weight='balanced',
        bootstrap=True,
        max_features='sqrt',
        oob_score=True,  # 启用袋外分数
        verbose=0
    ),
    'GradientBoosting': GradientBoostingClassifier(
        random_state=42,
        n_estimators=1500,  # 显著增加树的数量
        learning_rate=0.02,  # 进一步降低学习率
        max_depth=7,  # 调整树的深度，平衡偏差和方差
        min_samples_split=5,  # 调整分裂样本数
        min_samples_leaf=3,  # 调整叶子节点样本数
        subsample=0.85,  # 调整子样本比例
        max_features='sqrt',  # 特征选择策略
        verbose=0
    ),
    'LightGBM': LGBMClassifier(
        random_state=42,
        n_estimators=1000,  # 增加树的数量
        learning_rate=0.03,  # 降低学习率
        max_depth=10,  # 增加树的深度
        num_leaves=128,  # 增加叶子节点数量
        subsample=0.9,  # 增加子样本比例
        colsample_bytree=0.9,  # 增加列采样比例
        reg_lambda=5,  # 增加L2正则化
        reg_alpha=1,  # 增加L1正则化
        class_weight='balanced',
        verbose=-1,
        boosting_type='gbdt',
        objective='binary',
        metric='binary_logloss'
    ),
    'XGBoost': XGBClassifier(
        random_state=42,
        n_estimators=1000,  # 增加树的数量
        learning_rate=0.03,  # 降低学习率
        max_depth=7,  # 树的深度
        min_child_weight=3,  # 子节点最小权重
        subsample=0.85,  # 子样本比例
        colsample_bytree=0.85,  # 列采样比例
        reg_alpha=1,  # L1正则化
        reg_lambda=5,  # L2正则化
        gamma=0,  # 节点分裂的最小损失减少
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    ),
    'CatBoost': CatBoostClassifier(
        random_state=42,
        iterations=500,  # 减少迭代次数，加快训练速度
        learning_rate=0.05,  # 提高学习率
        depth=7,  # 减少树的深度
        auto_class_weights='Balanced',
        subsample=0.8,  # 调整子样本比例
        colsample_bylevel=0.8,  # 调整列采样比例
        l2_leaf_reg=3,  # 调整L2正则化
        bootstrap_type='Bernoulli',
        grow_policy='SymmetricTree',  # 使用更简单的生长策略
        verbose=0
    )
}

# 训练和交叉验证所有模型
model_results = {}
print("\n" + "="*60)
for name, model in models.items():
    print(f"\n训练并交叉验证 {name} 模型...")
    
    # 使用过采样后的数据训练模型
    model.fit(X_train_resampled, y_train_resampled)
    
    # 交叉验证评估 - 使用原始数据（不过采样）进行评估，更接近真实情况
    cv_scores = cross_val_score(
        model, 
        X_train_selected_df, 
        y_train, 
        cv=cv, 
        scoring='accuracy', 
        n_jobs=-1
    )
    
    # 存储结果
    model_results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    # 只显示交叉验证结果
    print(f"{name} 交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"各折结果: {[round(score, 4) for score in cv_scores]}")

print("\n" + "="*60)

# 训练集成模型 - 只使用表现最好的两个模型，避免XGBoost的兼容性问题
print("\n训练并交叉验证集成模型...")

# 只使用两个表现最好的模型（GradientBoosting和RandomForest）
base_estimators = [
    ('gb', GradientBoostingClassifier(
        random_state=42,
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.85,
        max_features='sqrt',
        verbose=0
    )),
    ('rf', RandomForestClassifier(
        random_state=42,
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        bootstrap=True,
        max_features='sqrt',
        oob_score=True,
        verbose=0
    ))
]

# 使用Voting Ensemble，训练时间较短，效果较好
voting_model = VotingClassifier(
    estimators=base_estimators,
    voting='soft',
    n_jobs=-1,
    verbose=0
)

voting_model.fit(X_train_resampled, y_train_resampled)
voting_cv_scores = cross_val_score(
    voting_model, 
    X_train_selected_df, 
    y_train, 
    cv=cv, 
    scoring='accuracy', 
    n_jobs=-1
)

model_results['Voting'] = {
    'model': voting_model,
    'cv_mean': voting_cv_scores.mean(),
    'cv_std': voting_cv_scores.std(),
    'cv_scores': voting_cv_scores
}

print(f"Voting Ensemble 交叉验证准确率: {voting_cv_scores.mean():.4f} ± {voting_cv_scores.std():.4f}")
print(f"各折结果: {[round(score, 4) for score in voting_cv_scores]}")

# 显示最佳模型
print("\n" + "="*60)
print("所有模型交叉验证结果汇总:")
for name, result in model_results.items():
    print(f"{name}: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")

# 选择最佳模型
best_model_name = max(model_results, key=lambda x: model_results[x]['cv_mean'])
best_model = model_results[best_model_name]['model']
best_cv_mean = model_results[best_model_name]['cv_mean']
best_cv_std = model_results[best_model_name]['cv_std']

print(f"\n最佳模型: {best_model_name}")
print(f"最佳交叉验证准确率: {best_cv_mean:.4f} ± {best_cv_std:.4f}")

# 使用最佳模型进行预测
y_pred = best_model.predict(X_test_selected_df)

# 保存预测结果
submission = pd.DataFrame({'prediction': y_pred})
submission['prediction'] = submission['prediction'].map({1: 'yes', 0: 'no'})
subdir = ''
submission.to_csv(subdir + student_id + 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False)

print("\n预测结果已生成！")
print("\n注意: 所有结果均基于5折交叉验证，避免了过拟合风险")
