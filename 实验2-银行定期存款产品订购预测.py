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


student_id= '你的学号'
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


# 简单的数据预处理
def preprocess_data(df):
    data = df.copy()

    # 删除duration特征（会造成数据泄露）
    if 'duration' in data.columns:
        data = data.drop('duration', axis=1)

    # 处理分类变量 - 简单标签编码
    categorical_columns = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'poutcome', 'month'
    ]

    for col in categorical_columns:
        if col in data.columns:
            # 处理未知值
            if data[col].dtype == 'object':
                data[col] = data[col].replace('unknown', np.nan)
                # 用众数填充缺失值
                if data[col].isna().any():
                    mode_val = data[col].mode()
                    if len(mode_val) > 0:
                        data[col] = data[col].fillna(mode_val[0])

            # 标签编码
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

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
numeric_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 训练逻辑回归模型
print("开始训练逻辑回归模型...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

print("模型训练完成！")


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
