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
from catboost import CatBoostClassifier

# 引入不平衡数据处理库
from imblearn.over_sampling import ADASYN

# 引入特征选择
from sklearn.feature_selection import VarianceThreshold

# 确保所有模型都已正确导入
print("已导入的模型: LightGBM, CatBoost, RandomForest, GradientBoosting, SVM")

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

    # 基础特征工程
    data['balance_bin'] = pd.qcut(data['balance'], q=4, labels=False, duplicates='drop')
    data['total_contacts'] = data['campaign'] + data['previous']
    data['is_first_contact'] = (data['previous'] == 0).astype(int)
    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=False, right=False)
    data['pdays'] = np.where(data['pdays'] == -1, 0, data['pdays'])
    data['is_recent_contact'] = (data['pdays'] <= 30).astype(int)

    # 特征安全处理
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

# 特征选择 - 去除低方差特征
print("\n进行特征选择...")
var_threshold = VarianceThreshold(threshold=0.01)
X_train_selected = var_threshold.fit_transform(X_train)
X_test_selected = var_threshold.transform(X_test)
selected_cols = X_train.columns[var_threshold.get_support()]
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_cols, index=X_train.index)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_cols, index=X_test.index)
print(f"特征选择完成: {X_train.shape[1]} → {X_train_selected_df.shape[1]}")

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
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        bootstrap=True,
        max_features='sqrt'
    ),
    'GradientBoosting': GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.85,
        max_features='sqrt',
        verbose=0
    ),
    'LightGBM': LGBMClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.08,
        max_depth=7,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3,
        class_weight='balanced',
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        random_state=42,
        iterations=300,
        learning_rate=0.08,
        depth=7,
        auto_class_weights='Balanced',
        subsample=0.85,
        colsample_bylevel=0.85,
        l2_leaf_reg=3,
        bootstrap_type='Bernoulli',
        grow_policy='SymmetricTree',
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

# 训练集成模型
print("\n训练并交叉验证集成模型...")

# 准备基础模型
base_estimators = [
    ('rf', models['RandomForest']),
    ('gb', models['GradientBoosting']),
    ('lgbm', models['LightGBM']),
    ('catboost', models['CatBoost'])
]

# 1. Stacking Ensemble
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(
        random_state=42,
        max_iter=3000,
        C=2.0,
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced'
    ),
    cv=5,
    n_jobs=-1,
    passthrough=True
)

stacking_model.fit(X_train_resampled, y_train_resampled)
stacking_cv_scores = cross_val_score(
    stacking_model, 
    X_train_selected_df, 
    y_train, 
    cv=cv, 
    scoring='accuracy', 
    n_jobs=-1
)

model_results['Stacking'] = {
    'model': stacking_model,
    'cv_mean': stacking_cv_scores.mean(),
    'cv_std': stacking_cv_scores.std(),
    'cv_scores': stacking_cv_scores
}

print(f"Stacking Ensemble 交叉验证准确率: {stacking_cv_scores.mean():.4f} ± {stacking_cv_scores.std():.4f}")
print(f"各折结果: {[round(score, 4) for score in stacking_cv_scores]}")

# 2. Voting Ensemble
voting_model = VotingClassifier(
    estimators=base_estimators,
    voting='soft',
    n_jobs=-1
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
