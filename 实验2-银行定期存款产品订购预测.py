#!/usr/bin/env python
# coding: utf-8

# # 实验2-银行定期存款产品订购预测

# ## 学号：
# ## 姓名：
# ## 班级：

# ## **背景与业务目标**
# 
# 一家银行希望通过电话营销来推广其定期存款产品。然而，广泛的、无差别的营销成本高昂且客户体验不佳。营销部门希望数据科学团队能够构建一个预测模型，**精准识别哪些客户最有可能订阅定期存款产品**。
# 
# 这样，银行可以将营销资源集中在高潜力的客户群体上，从而提高营销效率、降低成本并减少对低意向客户的打扰。
# 
# ### **数据说明**
# 
# 您获得的数据集包含了银行在一次营销活动后记录的客户信息与结果（见您提供的截图）。数据包含以下主要字段：
# 
# - **特征变量 (Features):**
# - `age`: 年龄（数值型）
# - `job`: 职业类型（分类变量，如：admin., technician, management, blue-collar等）
# - `marital`: 婚姻状况（分类变量：married, single, divorced）
# - `education`: 教育程度（分类变量：primary, secondary, tertiary, unknown）
# - `default`: 是否有违约记录（二分变量：yes, no）
# - `balance`: 年均账户余额（数值型，单位可能是欧元或美元）
# - `housing`: 是否有住房贷款（二分变量：yes, no）
# - `loan`: 是否有个人贷款（二分变量：yes, no）
# - `contact`: 联系方式（分类变量：unknown, cellular, telephone）
# - `day`, `month`: 最后一次联系的时间
# - `duration`: 最后一次通话的持续时间（秒，**重要提示：该特征在预测时不可用**，因为只有在通话结束后才知道时长）
# - `campaign`: 本次活动中与该客户联系的次数
# - `pdays`: 从上一次营销活动后经过的天数（-1表示此前未联系过）
# - `previous`: 在本次活动之前与该客户联系的次数
# - `poutcome`: 上一次营销活动的结果（分类变量：unknown, failure, success, other）
# - **目标变量 (Target Variable):**
# - `deposit`: 客户是否订阅了定期存款（二分变量：**yes, no**）
# 
# ## **任务要求**
# 
# 请完成以下步骤，构建一个二分类预测模型：
# 
# 
# 本道题目完成时间为四节课。
# 
# 首先提供测试集A（文件名：testA.csv），
# 
# 第四节课提供测试集B（文件名：testB.csv），
# 
# 最终排名以测试集B的分数为准。

# ## 测评标准
# 准确率
# 
# $$
# (TP+TN)/(TP+FP+FN+TN)$$
# 总体预测正确的比例
# 
# 

# ### 输入学号

# In[1]:


student_id= '25451354008'
#输入学号，不输入或者输错则没有成绩


# In[2]:


# === 1. 导入必要的库 ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
import datetime
import warnings
warnings.filterwarnings('ignore')


# ## 读取数据

# In[3]:


# path to where the data lies
dpath = 'data/'
train=pd.read_csv(dpath +"train.csv")
test=pd.read_csv(dpath +"testA.csv")


# In[4]:


## 输出数据的大小信息
print('Train data shape:',train.shape)
print('TestA data shape:',test.shape)
print("\n目标变量 'deposit' 的分布:")
print(train['deposit'].value_counts())


# In[5]:


train.head()


# ## 数据可视化和探索部分

# In[6]:


#建议对数据可视化和探索来发现有用的特征


# ## 数据建模部分

# In[7]:


# 改进的数据预处理
def preprocess_data(df):
    data = df.copy()

    # 删除duration特征（会造成数据泄露）
    if 'duration' in data.columns:
        data = data.drop('duration', axis=1)

    # 处理分类变量 - 使用更合适的编码方式
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
    data['pdays_category'] = np.where(data['pdays'] == -1, 'no_contact', 
                                     np.where(data['pdays'] <= 30, 'recent', 
                                             np.where(data['pdays'] <= 90, 'medium', 'old')))
    data = pd.get_dummies(data, columns=['pdays_category'], drop_first=True)
    
    # 3. pdays的数值特征处理
    data['pdays'] = np.where(data['pdays'] == -1, 0, data['pdays'])  # 处理-1值

    # 4. 联系次数的衍生特征
    data['total_contacts'] = data['campaign'] + data['previous']
    data['is_first_contact'] = (data['previous'] == 0).astype(int)  # 是否首次联系

    # 5. 年龄的分箱特征
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 100], labels=False, right=False)

    # 6. 月的季节特征 - 简化处理
    # 直接从原始数据获取月份信息（在独热编码前）
    if 'month' in df.columns:  # 只在训练集处理，测试集已经处理过
        original_month = df['month'].copy()
        # 创建月份映射
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        season_map = {
            1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall',
            11: 'fall', 12: 'winter'
        }
        # 添加季节特征
        data['season'] = original_month.map(month_map).map(season_map)
        data = pd.get_dummies(data, columns=['season'], drop_first=True)

    return data


# 预处理训练集和测试集
train_processed = preprocess_data(train)
test_processed = preprocess_data(test)

# 尝试多种模型
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# 引入先进的梯度提升模型
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    import xgboost as xgb
    from xgboost import XGBClassifier
    advanced_models_available = True
except ImportError:
    print("LightGBM或XGBoost未安装，将使用传统模型")
    advanced_models_available = False

# 模型列表（基本模型）
basic_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.1, max_depth=5)
}

# 准备特征和目标变量
X_train = train_processed.drop('deposit', axis=1)
y_train = train_processed['deposit'].map({'yes': 1, 'no': 0})

# 测试集特征
X_test = test_processed

# 数值特征标准化 - 更新数值特征列表
numeric_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous', 
                'total_contacts', 'is_first_contact', 'age_group', 'balance_bin']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 添加先进模型（如果可用）
models = basic_models.copy()
if advanced_models_available:
    models['XGBoost'] = XGBClassifier(
        random_state=42, 
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # 处理不平衡数据
    )
    models['LightGBM'] = LGBMClassifier(
        random_state=42, 
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        class_weight='balanced',
        verbose=-1
    )

# 添加集成模型
if len(models) >= 2:
    voting_estimators = [(name, model) for name, model in models.items()]
    models['Voting Ensemble'] = VotingClassifier(
        estimators=voting_estimators, 
        voting='soft', 
        n_jobs=-1
    )

# 交叉验证评估模型
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("开始模型训练和交叉验证...")
for name, model in models.items():
    print(f"\n训练 {name}...")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f"{name} 交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# 尝试使用Stacking Ensemble提高性能
print("\n尝试使用Stacking Ensemble...")
try:
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression as StackingMetaModel
    
    # 准备基础模型
    base_estimators = [
        ('rf', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(random_state=42, n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8)),
        ('xgb', XGBClassifier(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])))
    ]
    
    # 创建Stacking模型
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=StackingMetaModel(random_state=42, max_iter=2000, class_weight='balanced'),
        cv=cv,
        n_jobs=-1
    )
    
    # 训练和评估Stacking模型
    stacking_score = cross_val_score(stacking_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Stacking Ensemble 交叉验证准确率: {stacking_score.mean():.4f} ± {stacking_score.std():.4f}")
    
    # 如果Stacking模型表现更好，将其添加到模型列表
    models['Stacking Ensemble'] = stacking_model
    results['Stacking Ensemble'] = stacking_score
except Exception as e:
    print(f"Stacking Ensemble训练失败: {e}")

# 选择最优模型
best_model_name = max(results, key=lambda k: results[k].mean())
best_model_score = results[best_model_name].mean()
print(f"\n最佳模型: {best_model_name}, 准确率: {best_model_score:.4f}")

# 对表现最好的模型进行进一步优化
print(f"\n对{best_model_name}进行进一步优化...")

# 使用经过验证的良好参数，避免长时间的网格搜索
if best_model_name == 'Gradient Boosting':
    # 使用经过验证的最优参数
    best_model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=400,
        learning_rate=0.08,
        max_depth=7,
        subsample=0.9,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    # 训练模型
    best_model.fit(X_train, y_train)
    
    # 使用交叉验证评估最终模型
    final_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"优化后Gradient Boosting交叉验证准确率: {final_scores.mean():.4f} ± {final_scores.std():.4f}")
    
    # 添加特征选择，基于特征重要性
    print("\n进行特征选择...")
    from sklearn.feature_selection import SelectFromModel
    
    # 基于Gradient Boosting模型的特征重要性进行选择
    selector = SelectFromModel(best_model, threshold='median', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"原始特征数量: {X_train.shape[1]}, 选择后特征数量: {X_train_selected.shape[1]}")
    
    # 使用选择后的特征重新训练模型
    best_model_selected = GradientBoostingClassifier(
        random_state=42,
        n_estimators=400,
        learning_rate=0.08,
        max_depth=7,
        subsample=0.9,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    best_model_selected.fit(X_train_selected, y_train)
    
    # 评估特征选择后的模型
    selected_scores = cross_val_score(best_model_selected, X_train_selected, y_train, cv=cv, scoring='accuracy')
    print(f"特征选择后模型交叉验证准确率: {selected_scores.mean():.4f} ± {selected_scores.std():.4f}")
    
    # 如果特征选择后的模型表现更好，使用它
    if selected_scores.mean() > final_scores.mean():
        best_model = best_model_selected
        X_test = X_test_selected
        print("使用特征选择后的模型")
    else:
        print("特征选择没有提高性能，继续使用原始模型")
else:
    # 使用交叉验证中表现最好的模型
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

# 特征重要性分析
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    if 'X_train_selected' in locals() and X_train_selected.shape[1] != X_train.shape[1]:
        # 如果进行了特征选择，只显示选择后的特征
        print("\n特征选择后的特征重要性（前10名）：")
        print("注意：由于进行了特征选择，无法显示具体特征名称")
        print(f"特征重要性列表：{importances[:10]}")
    else:
        # 否则显示完整特征重要性
        feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        print("\n特征重要性排序（前10名）：")
        print(feature_importance_df.head(10))

# 预测
y_pred = best_model.predict(X_test)

print("\n模型训练完成！")
print(f"最终模型准确率: {best_model_score:.4f}")


# ## 提交结果文件
# 这里不用改动，自动会记录提交的时间

# In[8]:


# 保存预测结果
submission = pd.DataFrame({'prediction': y_pred})
submission['prediction'] = submission['prediction'].map({1: 'yes', 0: 'no'})
subdir = ''
submission.to_csv(subdir + student_id + 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False)


# 最后把csv文件上传到ftp://10.132.219.5:955 的相应目录
