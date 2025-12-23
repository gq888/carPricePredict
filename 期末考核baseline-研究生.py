#!/usr/bin/env python
# coding: utf-8

# # 期末考核大作业

# ## 学号：
# ## 姓名：
# ## 班级：

# ### 考核题目：基于多维度特征的贷款违约风险预测
# 
# **题目描述**：本题目要求基于用户的个人信息、信用历史、贷款属性等多维度特征，预测贷款是否会发生违约。
# 
# 本题目重点考察数据预处理能力、特征工程技巧、模型选择与优化能力，以及结果解释能力。建议在建模过程中特别注意类别不平衡问题的处理，并注重模型的可解释性。
# 
# ### 考核时间安排
# 
# 本道题目测评A版完成时间为**2025年12月30日**22点。
# 
# A版（文件名：testA.csv），每5分钟进行一次测评，**2025年12月30日星期一上午8点半**提供测试集B（文件名：testB.csv），在10/12/16/22点进行4次测评，**2025年12月30日**22点。
# 
# 最终排名以测试集B的分数为准。
# 
# （注意：只有A榜成绩没有B榜成绩，或者只有B榜成绩没有A榜成绩，排名成绩为第四段最后）
# 
# ### 测评标准
# 
# -   准确率（Accuracy）
# 
# -   **重要要求**：不得使用外来数据
# 
# ### 提交文件格式
# 
# 您必须提交一个带有列表`标签`的csv文件，包含两列：**贷款ID**和**是否违约**。
# 
# ### 数据特征描述
# 
# 以下是主要的特征变量说明：
# 
# | 字段名称                | 详细说明                                         | 数据类型        | 备注                                 |
# |-------------------------|--------------------------------------------------|-----------------|--------------------------------------|
# | **贷款ID**              | 为每笔贷款分配的唯一标识符                       | 离散型          | 主键，用于唯一识别每条贷款记录       |
# | **用户ID**              | 贷款申请人的唯一标识符                           | 离散型          | 用于关联同一用户的多笔贷款           |
# | **贷款总额**            | 申请的贷款总金额                                 | 连续型          | 重要风控指标，单位通常为元           |
# | **贷款年限**            | 贷款的期限，以年为单位                           | 离散型          | 例如 3年或5年                        |
# | **利率**                | 贷款的年化利率                                   | 连续型          | 百分比，反映资金成本和个人信用状况   |
# | **月还款额**            | 根据贷款总额、年限和利率计算出的每月偿还金额     | 连续型          | 重要风控指标，单位通常为元           |
# | **信用等级**            | 内部评定的借款人信用等级（如A, B, C）            | 分类型          | 等级越高，信用风险通常越低           |
# | **雇主类型**            | 借款人所在单位的类型（如上市公司、政府机构等）   | 分类型          | 反映工作稳定性和收入质量             |
# | **所属行业**            | 借款人从事的行业领域（如金融、IT、教育等）       | 分类型          | 不同行业的风险特征不同               |
# | **工作年限**            | 借款人的工作时间长度                             | 离散型          | 通常与收入稳定性正相关               |
# | **是否有房**            | 标识借款人是否拥有房产                           | 二分类型（0/1） | 是重要的资产证明，0=无，1=有         |
# | **审核状态**            | 贷款申请的审核状态                               | 分类型          | 如通过、拒绝、待审核等               |
# | **贷款用途**            | 借款人申明的贷款资金用途（如消费、教育、经营等） | 分类型          | 不同用途的违约风险不同               |
# | **邮政编码**            | 借款人提供的邮政编码前几位                       | 分类型          | 脱敏后的地区信息                     |
# | **所在地区**            | 借款人所在的地区编码                             | 分类型          | 用于地域风险分析                     |
# | **负债收入比**          | 月度债务支出与月收入的比率                       | 连续型          | **核心风控指标**，值越高风险越大     |
# | **18个月内逾期次数**    | 过去18个月内信用档案中发生逾期的次数             | 离散型          | 直接反映历史信用行为                 |
# | **信用评分低值**        | 信用评分区间的下限                               | 连续型          | 与信用评分高值共同定义信用分范围     |
# | **信用评分高值**        | 信用评分区间的上限                               | 连续型          |                                      |
# | **公共记录破产数**      | 在公共记录中存在的破产记录数量                   | 离散型          | 严重的负面信用信息                   |
# | **循环信用额度B/U**     | 循环信用账户的相关额度                           | 连续型          | 反映借款人的信用额度使用情况         |
# | **初始列表状态**        | 贷款信息初始发布时的状态                         | 分类型          |                                      |
# | **最早信用账户开通月**  | 借款人报告的最早信用账户开通的时间               | 日期型（文本）  | 用于计算信用历史长度                 |
# | **职称**                | 借款人的职位或职称                               | 分类型          | 间接反映收入和社会地位               |
# | **策略代码**            | 内部策略代码                                     | 分类型          |                                      |
# | **特征0-特征4**         | 经过处理的匿名特征                               | 混合类型        | 通常为贷款人行为计数特征等           |
# | **是否提前还款**        | 标识该笔贷款是否有提前还款行为                   | 二分类型（0/1） | 0=否，1=是                           |
# | **提前还款金额**        | 累计提前偿还的本金金额                           | 连续型          |                                      |
# | **近3个月提前还款金额** | 最近三个月内的提前还款金额                       | 连续型          | 反映近期资金流动和还款意愿           |
# | **贷款发放年份/月份**   | 贷款发放的具体年份和月份                         | 离散型          | 用于时间序列分析                     |
# | **贷款发放日期差**      | 可能与基准日期的时间差                           | 离散型          |                                      |
# | **是否违约**            | **目标变量**，标识贷款是否最终违约               | 二分类型（0/1） | **建模预测的目标**，0=未违约，1=违约 |
# 
# **目标变量**：
# 
# **是否违约**
# 
# -   **贷款正常**：0
# 
# -   **贷款违约**：1
# 
# ### 数据挖掘报告要求
# 
# 作业报告基于B榜最优分数编写，用中文完成（可附带英文版本），整合在Jupyter
# notebook文件中，至少包括以下部分：
# 
# -   数据清洗
# 
# -   探索性数据分析(EDA)
# 
# -   特征工程，通常包括特征工程和特征选择
# 
# -   数据建模，通常包括基于性能指标比较几种机器学习模型、对最佳模型执行超参数调整、在测试集上评估最佳模型、解释模型结果
# 
# -   陈述总结
# 
# -   参考文献，注意全部列出的参考文献需在文中引用。
# 
# ### 评分细则
# 
# -   **期末大作业成绩** = 最终B榜排名（70%） + 作业报告（30%）
# 
# -   **作业报告** = 格式（20%） + 内容（80%）
# 
# -   **课程总成绩**：期末大作业成绩（60%） + 平时成绩（40%）
# 
# ### 上交文件要求
# 
# 1.  **作业代码和报告**：整合成可运行的Jupyter
#     notebook文件上交，同时另存为html或pdf文件 文件命名格式：`学号姓名期末大作业` 
#     
#     最后把文件上传到ftp://10.132.219.5:955 的相应目录
#     
#     上交截止时间：2026年1月5日晚上10点

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# In[2]:


train = pd.read_csv('data/train.csv')  # 训练集
test = pd.read_csv('data/testA.csv')   # 测试集


# ## 数据清洗

# In[ ]:





# ## 探索性数据分析(EDA)

# In[ ]:





# ## 特征工程

# In[5]:


# 2. 选择基础特征（数值型和重要类别型）
basic_features = [
    '贷款总额', '利率', '月还款额', '负债收入比', '18个月内逾期次数', '信用评分低值', '信用评分高值', '信用等级',
    '雇主类型', '是否有房'
]

# 3. 数据预处理
# 处理训练集
X_train = train[basic_features].copy()
y_train = train['是否违约']

# 处理测试集A
X_test = test[basic_features].copy()

# 4. 处理分类变量（简单标签编码）
categorical_cols = ['信用等级', '雇主类型', '是否有房']
for col in categorical_cols:
    # 合并数据确保编码一致
    all_data = pd.concat([X_train[col], X_test[col]], axis=0)
    le = LabelEncoder()
    le.fit(all_data.fillna('Unknown'))

    X_train[col + '_enc'] = le.transform(X_train[col].fillna('Unknown'))
    X_test[col + '_enc'] = le.transform(X_test[col].fillna('Unknown'))

# 删除原始分类列
X_train = X_train.drop(columns=categorical_cols)
X_test = X_test.drop(columns=categorical_cols)

# 5. 处理缺失值（简单填充0）
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# 6. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. 训练逻辑回归模型
print("训练逻辑回归模型中...")
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# 8. 在测试集A上预测
test_predictions = logreg.predict(X_test_scaled)
test_probabilities = logreg.predict_proba(X_test_scaled)[:, 1]


# In[7]:


# 9. 生成提交文件
submission = pd.DataFrame({
    '贷款ID': test['贷款ID'],
    '是否违约': test_predictions  # 或者使用 test_probabilities 作为违约概率
})


# In[10]:


StudentId = '你的学号'#写自己的学号
subdir=''


# In[13]:


submission.to_csv(subdir + StudentId + 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False)


# ## 数据建模

# In[ ]:





# ## 陈述总结

# In[ ]:





# ## 参考文献

# In[ ]:




