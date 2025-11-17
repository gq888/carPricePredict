import pandas as pd
import numpy as np

# 读取数据
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/testA.csv')

print("=== 日期转换分析 ===")

# 1. 检查regDate的基本信息
print("\n1. regDate基本信息:")
print(f"数据类型: {train['regDate'].dtype}")
print(f"总样本数: {len(train['regDate'])}")
print(f"唯一值数量: {train['regDate'].nunique()}")

# 2. 检查regDate转换失败的情况
train['regDate_str'] = train['regDate'].astype(str)
train['regDate_dt'] = pd.to_datetime(train['regDate_str'], format='%Y%m%d', errors='coerce')

print("\n2. regDate转换情况:")
print(f"转换成功: {train['regDate_dt'].notna().sum()}")
print(f"转换失败: {train['regDate_dt'].isna().sum()}")

# 3. 查看转换失败的样本
if train['regDate_dt'].isna().sum() > 0:
    print("\n3. 转换失败的regDate示例:")
    failed_regDate = train[train['regDate_dt'].isna()]['regDate_str'].head(10)
    print(failed_regDate)

# 4. 检查字符串长度
print("\n4. regDate字符串长度分布:")
length_counts = train['regDate_str'].str.len().value_counts()
print(length_counts)

# 5. 检查creatDate的情况
print("\n5. creatDate转换情况:")
train['creatDate_str'] = train['creatDate'].astype(str)
train['creatDate_dt'] = pd.to_datetime(train['creatDate_str'], format='%Y%m%d', errors='coerce')
print(f"转换成功: {train['creatDate_dt'].notna().sum()}")
print(f"转换失败: {train['creatDate_dt'].isna().sum()}")

if train['creatDate_dt'].isna().sum() > 0:
    print("\n转换失败的creatDate示例:")
    failed_creatDate = train[train['creatDate_dt'].isna()]['creatDate_str'].head(10)
    print(failed_creatDate)

# 6. 检查是否有无效日期
print("\n6. 检查无效日期:")
# 尝试转换为字符串后再转换
train['regDate_str_padded'] = train['regDate_str'].str.pad(width=8, side='left', fillchar='0')
train['regDate_dt_padded'] = pd.to_datetime(train['regDate_str_padded'], format='%Y%m%d', errors='coerce')
print(f"填充后转换成功: {train['regDate_dt_padded'].notna().sum()}")
print(f"填充后转换失败: {train['regDate_dt_padded'].isna().sum()}")

# 7. 查看填充后仍转换失败的样本
if train['regDate_dt_padded'].isna().sum() > 0:
    print("\n7. 填充后仍转换失败的regDate示例:")
    failed_after_pad = train[train['regDate_dt_padded'].isna()][['regDate_str', 'regDate_str_padded']].head(10)
    print(failed_after_pad)