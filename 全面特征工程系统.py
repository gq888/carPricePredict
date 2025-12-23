#!/usr/bin/env python
# coding: utf-8

"""
全面特征工程系统
实现最完整和系统的特征工程处理流程，不计成本最大化模型性能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    LabelEncoder, OneHotEncoder, TargetEncoder, QuantileTransformer,
    PowerTransformer, FunctionTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, chi2,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox, yeojohnson, skew, kurtosis
import warnings
import joblib
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import re
import gc

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveFeatureEngineering:
    """全面特征工程系统类"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fitted_transformers = {}
        self.feature_info = {}
        self.categorical_features = []
        self.numerical_features = []
        self.datetime_features = []
        self.target = None
        
    def comprehensive_data_cleaning(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        全面数据清洗
        """
        print("=== 全面数据清洗阶段 ===")
        df_clean = df.copy()
        original_shape = df_clean.shape
        
        # 1. 处理重复数据
        print("1. 处理重复数据...")
        duplicates_before = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        duplicates_after = df_clean.duplicated().sum()
        print(f"   删除重复行: {duplicates_before - duplicates_after} 行")
        
        # 2. 识别特征类型
        print("2. 识别特征类型...")
        self._identify_feature_types(df_clean, target_col)
        
        # 3. 处理缺失值 - 多种策略
        print("3. 处理缺失值...")
        df_clean = self._advanced_missing_value_imputation(df_clean)
        
        # 4. 处理异常值 - 多种检测方法
        print("4. 处理异常值...")
        df_clean = self._comprehensive_outlier_detection(df_clean)
        
        # 5. 数据类型优化
        print("5. 数据类型优化...")
        df_clean = self._optimize_data_types(df_clean)
        
        print(f"数据清洗完成: {original_shape} -> {df_clean.shape}")
        return df_clean
    
    def _identify_feature_types(self, df: pd.DataFrame, target_col: str = None):
        """识别特征类型"""
        for col in df.columns:
            if target_col and col == target_col:
                continue
                
            if df[col].dtype == 'object':
                # 检查是否为日期
                try:
                    pd.to_datetime(df[col])
                    self.datetime_features.append(col)
                except:
                    # 检查是否为分类变量
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.1 or df[col].nunique() < 20:
                        self.categorical_features.append(col)
                    else:
                        self.numerical_features.append(col)
            elif np.issubdtype(df[col].dtype, np.datetime64):
                self.datetime_features.append(col)
            elif np.issubdtype(df[col].dtype, np.number):
                self.numerical_features.append(col)
        
        print(f"   数值特征: {len(self.numerical_features)} 个")
        print(f"   分类特征: {len(self.categorical_features)} 个")
        print(f"   时间特征: {len(self.datetime_features)} 个")
    
    def _advanced_missing_value_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级缺失值处理"""
        df_imputed = df.copy()
        
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
                
            missing_ratio = df[col].isnull().sum() / len(df)
            print(f"   {col}: 缺失率 {missing_ratio:.3f}")
            
            if col in self.numerical_features:
                if missing_ratio < 0.1:
                    # 低缺失率 - 使用均值/中位数
                    if abs(skew(df[col].dropna())) > 1:
                        df_imputed[col].fillna(df[col].median(), inplace=True)
                    else:
                        df_imputed[col].fillna(df[col].mean(), inplace=True)
                elif missing_ratio < 0.3:
                    # 中等缺失率 - 使用KNN插补
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]])
                    self.fitted_transformers[f'{col}_knn_imputer'] = imputer
                else:
                    # 高缺失率 - 创建缺失指示器并使用中位数
                    df_imputed[f'{col}_was_missing'] = df[col].isnull()
                    df_imputed[col].fillna(df[col].median(), inplace=True)
                    
            elif col in self.categorical_features:
                if missing_ratio < 0.1:
                    # 低缺失率 - 使用众数
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df_imputed[col].fillna(mode_val, inplace=True)
                else:
                    # 高缺失率 - 创建新类别
                    df_imputed[col].fillna('Missing', inplace=True)
                    
        return df_imputed
    
    def _comprehensive_outlier_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """全面异常值检测"""
        df_clean = df.copy()
        
        for col in self.numerical_features:
            if col not in df.columns or df[col].dtype == 'object':
                continue
                
            # 多种异常值检测方法
            outliers_mask = np.zeros(len(df), dtype=bool)
            
            # 1. IQR方法
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # 2. Z-score方法
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_outliers = np.zeros(len(df), dtype=bool)
            z_outliers[df[col].notna()] = z_scores > 3
            
            # 3. 修正Z-score方法 (更鲁棒)
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            modified_z_scores = 0.6745 * (df[col] - median) / mad
            modified_z_outliers = np.abs(modified_z_scores) > 3.5
            
            # 4. 孤立森林 (对于高维数据)
            if len(df) > 1000:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
                iso_outliers = iso_forest.fit_predict(df[[col]]) == -1
            else:
                iso_outliers = np.zeros(len(df), dtype=bool)
            
            # 综合异常值判断 (使用投票机制)
            outlier_votes = iqr_outliers.astype(int) + z_outliers.astype(int) + \
                          modified_z_outliers.astype(int) + iso_outliers.astype(int)
            outliers_mask = outlier_votes >= 2  # 至少2种方法认为是异常值
            
            if outliers_mask.sum() > 0:
                print(f"   {col}: 检测到 {outliers_mask.sum()} 个异常值 ({outliers_mask.sum()/len(df)*100:.2f}%)")
                
                # 根据异常值比例决定处理方式
                outlier_ratio = outliers_mask.sum() / len(df)
                if outlier_ratio < 0.05:  # 少于5%的异常值可以删除
                    df_clean = df_clean[~outliers_mask]
                    print(f"     删除异常值: {outliers_mask.sum()} 个")
                else:  # 较多异常值进行截断处理
                    # 截断到合理范围
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    print(f"     截断异常值到合理范围")
        
        return df_clean
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型以节省内存"""
        df_optimized = df.copy()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object' and not np.issubdtype(col_type, np.datetime64):
                c_min = df[col].min()
                c_max = df[col].max()
                
                # 跳过时间戳类型
                if hasattr(c_min, 'year'):  # 检查是否为时间戳
                    continue
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df_optimized[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_optimized[col] = df[col].astype(np.float32)
        
        return df_optimized
    
    def comprehensive_feature_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全面特征转换
        """
        print("=== 全面特征转换阶段 ===")
        df_transformed = df.copy()
        
        # 1. 数值特征变换
        print("1. 数值特征变换...")
        df_transformed = self._advanced_numerical_transformations(df_transformed)
        
        # 2. 分类特征编码
        print("2. 分类特征编码...")
        df_transformed = self._advanced_categorical_encoding(df_transformed)
        
        # 3. 时间特征处理
        print("3. 时间特征处理...")
        df_transformed = self._comprehensive_datetime_features(df_transformed)
        
        # 4. 确保所有列都是数值类型 - 严格检查
        print("4. 数据类型验证...")
        for col in df_transformed.columns:
            if not pd.api.types.is_numeric_dtype(df_transformed[col]):
                try:
                    # 尝试转换为数值类型
                    df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')
                    if df_transformed[col].isna().all():
                        # 如果转换后全是NaN，删除该列
                        df_transformed = df_transformed.drop(columns=[col])
                        print(f"   删除无法转换的列: {col}")
                    else:
                        print(f"   转换列 {col} 为数值类型")
                except:
                    # 如果转换失败，删除该列
                    df_transformed = df_transformed.drop(columns=[col])
                    print(f"   删除无法转换的列: {col}")
        
        print(f"   特征转换完成，最终特征数: {df_transformed.shape[1]}")
        return df_transformed
    
    def _advanced_numerical_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级数值特征变换"""
        df_transformed = df.copy()
        
        for col in self.numerical_features:
            if col not in df.columns:
                continue
                
            col_data = df[col]
            
            # 跳过零值和负值较多的特征
            if (col_data <= 0).sum() > len(df) * 0.1:
                continue
            
            # 1. 对数变换
            if col_data.min() > 0:
                log_transformed = np.log1p(col_data)
                if abs(skew(col_data) - skew(log_transformed)) > 0.5:
                    df_transformed[f'{col}_log'] = log_transformed
            
            # 2. 平方根变换
            sqrt_transformed = np.sqrt(np.maximum(col_data, 0))
            if abs(skew(col_data) - skew(sqrt_transformed)) > 0.3:
                df_transformed[f'{col}_sqrt'] = sqrt_transformed
            
            # 3. Box-Cox变换
            if col_data.min() > 0:
                try:
                    boxcox_transformed, lambda_param = boxcox(col_data)
                    if abs(skew(col_data) - skew(boxcox_transformed)) > 0.5:
                        df_transformed[f'{col}_boxcox'] = boxcox_transformed
                        self.fitted_transformers[f'{col}_boxcox'] = lambda_param
                except:
                    pass
            
            # 4. Yeo-Johnson变换 (处理负值)
            try:
                yj_transformed, lambda_param = yeojohnson(col_data)
                if abs(skew(col_data) - skew(yj_transformed)) > 0.5:
                    df_transformed[f'{col}_yeojohnson'] = yj_transformed
                    self.fitted_transformers[f'{col}_yeojohnson'] = lambda_param
            except:
                pass
            
            # 5. 分位数变换 (转换为正态分布)
            try:
                quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
                quantile_transformed = quantile_transformer.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                df_transformed[f'{col}_quantile'] = quantile_transformed
                self.fitted_transformers[f'{col}_quantile'] = quantile_transformer
            except:
                pass
            
            # 6. 幂变换
            for power in [2, 3, 0.5]:
                power_transformed = np.power(col_data, power)
                if abs(skew(col_data) - skew(power_transformed)) > 0.3:
                    df_transformed[f'{col}_power_{power}'] = power_transformed
            
            # 7. 倒数变换
            if col_data.min() > 0:
                reciprocal_transformed = 1 / (col_data + 1e-8)
                df_transformed[f'{col}_reciprocal'] = reciprocal_transformed
            
            # 8. 特征分箱
            if col_data.nunique() > 10:
                # 等宽分箱
                for n_bins in [5, 10, 20]:
                    try:
                        bins = pd.cut(col_data, bins=n_bins, labels=False)
                        df_transformed[f'{col}_bin_{n_bins}'] = bins
                    except:
                        pass
                
                # 等频分箱
                for n_bins in [5, 10, 20]:
                    try:
                        bins = pd.qcut(col_data, q=n_bins, labels=False, duplicates='drop')
                        df_transformed[f'{col}_qbin_{n_bins}'] = bins
                    except:
                        pass
        
        print(f"   数值变换完成，新增 {df_transformed.shape[1] - df.shape[1]} 个特征")
        return df_transformed
    
    def _advanced_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级分类特征编码"""
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            if col not in df.columns:
                continue
                
            # 1. 标签编码
            le = LabelEncoder()
            df_encoded[f'{col}_label'] = le.fit_transform(df[col].astype(str))
            self.fitted_transformers[f'{col}_label_encoder'] = le
            
            # 2. 目标编码 (如果有目标变量且长度匹配)
            if self.target is not None and len(self.target) == len(df):
                try:
                    target_encoder = TargetEncoder(random_state=self.random_state)
                    df_encoded[f'{col}_target'] = target_encoder.fit_transform(df[[col]], self.target)
                    self.fitted_transformers[f'{col}_target_encoder'] = target_encoder
                except Exception as e:
                    print(f"   目标编码失败 {col}: {str(e)}")
            
            # 3. 频率编码
            freq_encoding = df[col].value_counts().to_dict()
            df_encoded[f'{col}_freq'] = df[col].map(freq_encoding)
            
            # 4. 计数编码
            count_encoding = df[col].value_counts().to_dict()
            df_encoded[f'{col}_count'] = df[col].map(count_encoding)
            
            # 5. 均值编码 (基于数值特征)
            for num_col in self.numerical_features:
                if num_col in df.columns and num_col != col:
                    try:
                        mean_encoding = df.groupby(col)[num_col].mean()
                        df_encoded[f'{col}_mean_{num_col}'] = df[col].map(mean_encoding)
                    except Exception as e:
                        print(f"   均值编码失败 {col}-{num_col}: {str(e)}")
            
            # 6. WOE编码 (如果有目标变量且长度匹配)
            if self.target is not None and len(self.target) == len(df):
                try:
                    df_encoded = self._woe_encoding(df_encoded, col)
                except Exception as e:
                    print(f"   WOE编码失败 {col}: {str(e)}")
            
            # 7. 独热编码 (对于低基数分类)
            if df[col].nunique() <= 10:
                try:
                    ohe = OneHotEncoder(sparse_output=False, drop='first')
                    ohe_features = ohe.fit_transform(df[[col]])
                    ohe_feature_names = [f'{col}_ohe_{i}' for i in range(ohe_features.shape[1])]
                    for i, feature_name in enumerate(ohe_feature_names):
                        df_encoded[feature_name] = ohe_features[:, i]
                    self.fitted_transformers[f'{col}_ohe'] = ohe
                except Exception as e:
                    print(f"   独热编码失败 {col}: {str(e)}")
            
            # 8. 哈希编码 (对于高基数分类)
            if df[col].nunique() > 100:
                try:
                    from sklearn.feature_extraction import FeatureHasher
                    hasher = FeatureHasher(n_features=10, input_type='string')
                    hashed_features = hasher.transform(df[col].astype(str)).toarray()
                    for i in range(hashed_features.shape[1]):
                        df_encoded[f'{col}_hash_{i}'] = hashed_features[:, i]
                except Exception as e:
                    print(f"   哈希编码失败 {col}: {str(e)}")
        
        print(f"   分类编码完成，新增 {df_encoded.shape[1] - df.shape[1]} 个特征")
        return df_encoded
    
    def _woe_encoding(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """WOE编码实现"""
        if self.target is None:
            return df
            
        df_woe = df.copy()
        
        # 计算每个类别的WOE值
        woe_dict = {}
        total_good = (self.target == 1).sum()
        total_bad = (self.target == 0).sum()
        
        for category in df[col].unique():
            category_mask = df[col] == category
            category_good = (self.target[category_mask] == 1).sum()
            category_bad = (self.target[category_mask] == 0).sum()
            
            # 避免除零和log(0)
            if category_bad > 0 and category_good > 0 and total_bad > 0 and total_good > 0:
                woe = np.log((category_good / total_good) / (category_bad / total_bad))
                woe_dict[category] = woe
            else:
                woe_dict[category] = 0
        
        df_woe[f'{col}_woe'] = df[col].map(woe_dict)
        return df_woe
    
    def _comprehensive_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全面时间特征提取"""
        df_datetime = df.copy()
        
        for col in self.datetime_features:
            if col not in df.columns:
                continue
                
            # 转换为datetime
            try:
                datetime_series = pd.to_datetime(df[col])
            except:
                continue
            
            # 基础时间特征
            df_datetime[f'{col}_year'] = datetime_series.dt.year
            df_datetime[f'{col}_month'] = datetime_series.dt.month
            df_datetime[f'{col}_day'] = datetime_series.dt.day
            df_datetime[f'{col}_weekday'] = datetime_series.dt.weekday
            df_datetime[f'{col}_weekofyear'] = datetime_series.dt.isocalendar().week
            df_datetime[f'{col}_quarter'] = datetime_series.dt.quarter
            
            # 高级时间特征
            df_datetime[f'{col}_dayofyear'] = datetime_series.dt.dayofyear
            df_datetime[f'{col}_days_in_month'] = datetime_series.dt.days_in_month
            df_datetime[f'{col}_is_month_start'] = datetime_series.dt.is_month_start.astype(int)
            df_datetime[f'{col}_is_month_end'] = datetime_series.dt.is_month_end.astype(int)
            df_datetime[f'{col}_is_quarter_start'] = datetime_series.dt.is_quarter_start.astype(int)
            df_datetime[f'{col}_is_quarter_end'] = datetime_series.dt.is_quarter_end.astype(int)
            df_datetime[f'{col}_is_year_start'] = datetime_series.dt.is_year_start.astype(int)
            df_datetime[f'{col}_is_year_end'] = datetime_series.dt.is_year_end.astype(int)
            
            # 周期性特征 (正弦/余弦变换)
            df_datetime[f'{col}_month_sin'] = np.sin(2 * np.pi * datetime_series.dt.month / 12)
            df_datetime[f'{col}_month_cos'] = np.cos(2 * np.pi * datetime_series.dt.month / 12)
            df_datetime[f'{col}_day_sin'] = np.sin(2 * np.pi * datetime_series.dt.day / 31)
            df_datetime[f'{col}_day_cos'] = np.cos(2 * np.pi * datetime_series.dt.day / 31)
            df_datetime[f'{col}_weekday_sin'] = np.sin(2 * np.pi * datetime_series.dt.weekday / 7)
            df_datetime[f'{col}_weekday_cos'] = np.cos(2 * np.pi * datetime_series.dt.weekday / 7)
            
            # 时间差特征 (如果有多个时间特征)
            if len(self.datetime_features) > 1:
                for other_col in self.datetime_features:
                    if other_col != col and other_col in df.columns:
                        try:
                            other_datetime = pd.to_datetime(df[other_col])
                            time_diff = (datetime_series - other_datetime).dt.days
                            df_datetime[f'{col}_{other_col}_diff_days'] = time_diff
                        except:
                            pass
        
        print(f"   时间特征提取完成，新增 {df_datetime.shape[1] - df.shape[1]} 个特征")
        return df_datetime
    
    def comprehensive_feature_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全面特征创建 - 基于领域知识
        """
        print("=== 全面特征创建阶段 ===")
        df_created = df.copy()
        
        # 1. 金融相关特征 (基于领域知识)
        print("1. 金融相关特征创建...")
        df_created = self._create_financial_features(df_created)
        
        # 2. 统计特征
        print("2. 统计特征创建...")
        df_created = self._create_statistical_features(df_created)
        
        # 3. 聚合特征
        print("3. 聚合特征创建...")
        df_created = self._create_aggregation_features(df_created)
        
        # 4. 交互特征
        print("4. 交互特征创建...")
        df_created = self._create_interaction_features(df_created)
        
        # 5. 高阶多项式特征
        print("5. 高阶多项式特征创建...")
        df_created = self._create_polynomial_features(df_created)
        
        return df_created
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建金融相关特征"""
        df_financial = df.copy()
        
        # 贷款相关特征
        loan_cols = [col for col in df.columns if '贷款' in col or 'loan' in col.lower()]
        if len(loan_cols) > 1:
            df_financial['总贷款金额'] = df[loan_cols].sum(axis=1)
            df_financial['平均贷款金额'] = df[loan_cols].mean(axis=1)
            df_financial['贷款金额标准差'] = df[loan_cols].std(axis=1)
        
        # 收入相关特征
        income_cols = [col for col in df.columns if '收入' in col or 'income' in col.lower()]
        if len(income_cols) > 1:
            df_financial['总收入'] = df[income_cols].sum(axis=1)
            df_financial['收入稳定性'] = df[income_cols].std(axis=1)
        
        # 债务相关特征
        debt_cols = [col for col in df.columns if '债务' in col or 'debt' in col.lower()]
        if len(debt_cols) > 1:
            df_financial['总债务'] = df[debt_cols].sum(axis=1)
            df_financial['债务比率'] = df_financial['总债务'] / (df_financial['总收入'] + 1)
        
        # 信用相关特征
        credit_cols = [col for col in df.columns if '信用' in col or 'credit' in col.lower()]
        if len(credit_cols) > 0:
            df_financial['平均信用评分'] = df[credit_cols].mean(axis=1)
            df_financial['信用评分变化'] = df[credit_cols].max(axis=1) - df[credit_cols].min(axis=1)
        
        # 风险相关特征
        if '贷款总额' in df.columns and '收入' in df.columns:
            df_financial['债务收入比'] = df['贷款总额'] / (df['收入'] + 1)
        
        if '月供' in df.columns and '收入' in df.columns:
            df_financial['月供收入比'] = df['月供'] / (df['收入'] + 1)
        
        # 风险评分
        risk_factors = []
        if '债务收入比' in df_financial.columns:
            risk_factors.append(df_financial['债务收入比'])
        if '月供收入比' in df_financial.columns:
            risk_factors.append(df_financial['月供收入比'])
        if '信用评分低值' in df.columns:
            risk_factors.append(1 / (df['信用评分低值'] + 1))
        
        if risk_factors:
            df_financial['综合风险评分'] = np.mean(risk_factors, axis=0)
        
        return df_financial
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建统计特征"""
        df_stats = df.copy()
        
        # 数值特征的统计量 - 确保只选择真正的数值列
        num_cols = []
        for col in self.numerical_features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                num_cols.append(col)
        
        if len(num_cols) > 1:  # 需要至少2个数值特征才能计算统计量
            # 基础统计量
            df_stats['数值特征均值'] = df[num_cols].mean(axis=1)
            df_stats['数值特征标准差'] = df[num_cols].std(axis=1)
            df_stats['数值特征最大值'] = df[num_cols].max(axis=1)
            df_stats['数值特征最小值'] = df[num_cols].min(axis=1)
            df_stats['数值特征中位数'] = df[num_cols].median(axis=1)
            
            # 高级统计量
            df_stats['数值特征偏度'] = df[num_cols].skew(axis=1)
            df_stats['数值特征峰度'] = df[num_cols].kurt(axis=1)
            
            # 分位数特征
            df_stats['数值特征Q1'] = df[num_cols].quantile(0.25, axis=1)
            df_stats['数值特征Q3'] = df[num_cols].quantile(0.75, axis=1)
            df_stats['数值特征IQR'] = df_stats['数值特征Q3'] - df_stats['数值特征Q1']
            
            # 异常值统计
            outlier_counts = np.zeros(len(df))
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
                outlier_counts += outliers.astype(int)
            
            df_stats['数值特征异常值计数'] = outlier_counts
        
        return df_stats
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建聚合特征"""
        df_agg = df.copy()
        
        # 按分类变量聚合 - 确保只使用数值列进行聚合
        valid_num_cols = []
        for col in self.numerical_features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid_num_cols.append(col)
        
        for cat_col in self.categorical_features:
            if cat_col not in df.columns or df[cat_col].nunique() > 50:  # 限制基数
                continue
            
            for num_col in valid_num_cols:
                # 计算各种聚合统计量
                agg_stats = df.groupby(cat_col)[num_col].agg([
                    'mean', 'std', 'min', 'max', 'median'
                ]).to_dict()
                
                for stat_name, stat_dict in agg_stats.items():
                    df_agg[f'{cat_col}_{num_col}_{stat_name}'] = df[cat_col].map(stat_dict)
        
        return df_agg
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        print("   重新创建交互特征...")
        
        # 获取数值特征和分类特征
        num_cols = [col for col in self.numerical_features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        cat_cols = [col for col in self.categorical_features if col in df.columns]
        
        # 限制特征数量以避免内存问题
        max_num_features = min(8, len(num_cols))
        max_cat_features = min(5, len(cat_cols))
        
        selected_num_cols = num_cols[:max_num_features]
        selected_cat_cols = cat_cols[:max_cat_features]
        
        print(f"     选择 {len(selected_num_cols)} 个数值特征和 {len(selected_cat_cols)} 个分类特征")
        
        # 1. 数值-数值交互
        if len(selected_num_cols) >= 2:
            print("     创建数值-数值交互特征...")
            for i in range(len(selected_num_cols)):
                for j in range(i + 1, len(selected_num_cols)):
                    col1, col2 = selected_num_cols[i], selected_num_cols[j]
                    
                    # 基本运算
                    df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                    df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                    df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                    
                    # 避免除零
                    if df[col2].abs().min() > 1e-8:
                        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    if df[col1].abs().min() > 1e-8:
                        df[f'{col2}_div_{col1}'] = df[col2] / (df[col1] + 1e-8)
        
        # 2. 数值-分类交互
        if len(selected_num_cols) > 0 and len(selected_cat_cols) > 0:
            print("     创建数值-分类交互特征...")
            for num_col in selected_num_cols[:3]:  # 限制数量
                for cat_col in selected_cat_cols[:2]:  # 限制数量
                    # 按分类变量分组计算数值特征的统计量
                    group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'min', 'max', 'median'])
                    
                    # 创建新特征
                    for stat in ['mean', 'std', 'min', 'max', 'median']:
                        if stat in group_stats.columns:
                            df[f'{num_col}_{cat_col}_{stat}'] = df[cat_col].map(group_stats[stat])
        
        print(f"     交互特征创建完成，总共新增 {df.shape[1] - len(num_cols) - len(cat_cols)} 个特征")
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建多项式特征"""
        print("   重新创建多项式特征...")
        
        # 确保只使用数值列
        valid_num_cols = []
        for col in df.columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid_num_cols.append(col)
        
        if len(valid_num_cols) < 2:
            print(f"     数值特征不足，跳过多项式特征创建")
            return df
        
        # 限制特征数量以避免内存问题
        max_features = min(10, len(valid_num_cols))  # 限制为最多10个特征
        selected_cols = valid_num_cols[:max_features]
        
        print(f"     选择 {len(selected_cols)} 个特征进行多项式变换")
        
        # 填充缺失值
        df_filled = df[selected_cols].fillna(0)
        
        try:
            # 创建多项式特征 (degree=2)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(df_filled)
            
            # 获取特征名称
            feature_names = poly.get_feature_names_out(selected_cols)
            poly_df = pd.DataFrame(poly_features, columns=feature_names)
            
            # 基于方差选择重要的多项式特征 (限制数量)
            selected_features = []
            
            for poly_col in feature_names:
                # 跳过原始特征
                if ' ' not in poly_col:  # 单个特征
                    continue
                
                # 计算方差
                variance = poly_df[poly_col].var()
                if variance > 0.01:  # 方差阈值
                    selected_features.append(poly_col)
            
            # 限制特征数量
            max_poly_features = min(20, len(selected_features))
            final_features = selected_features[:max_poly_features]
            
            if len(final_features) > 0:
                # 添加选中的多项式特征
                for feature in final_features:
                    df[f'poly_{feature}'] = poly_df[feature]
                
                self.fitted_transformers['polynomial'] = poly
                print(f"     创建 {len(final_features)} 个多项式特征")
            else:
                print(f"     没有创建多项式特征")
                
        except Exception as e:
            print(f"     多项式特征创建失败: {e}")
        
        return df
    
    def comprehensive_feature_selection(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        全面特征选择 - 多种方法结合
        """
        print("=== 全面特征选择阶段 ===")
        df_selected = df.copy()
        
        # 确保所有列都是数值类型且没有无穷值
        print("1. 数据预处理...")
        numeric_cols = []
        for col in df_selected.columns:
            if pd.api.types.is_numeric_dtype(df_selected[col]):
                # 处理无穷值
                df_selected[col] = df_selected[col].replace([np.inf, -np.inf], np.nan)
                df_selected[col] = df_selected[col].fillna(df_selected[col].median())
                numeric_cols.append(col)
            else:
                print(f"   删除非数值列: {col}")
                df_selected = df_selected.drop(columns=[col])
        
        print(f"   数值特征数量: {len(numeric_cols)}")
        
        if len(numeric_cols) == 0:
            print("   警告: 没有数值特征可用于选择")
            return df_selected
        
        # 1. 方差选择法
        print("2. 方差选择法...")
        try:
            variance_selector = VarianceThreshold(threshold=0.01)
            df_variance = variance_selector.fit_transform(df_selected[numeric_cols])
            variance_selected = [numeric_cols[i] for i in variance_selector.get_support(indices=True)]
            print(f"   方差选择法: {len(variance_selected)} 个特征")
        except Exception as e:
            print(f"   方差选择法失败: {e}")
            variance_selected = numeric_cols
        
        # 2. 统计检验法
        print("3. 统计检验法...")
        try:
            # 确保目标变量是数值类型
            target_numeric = pd.to_numeric(target, errors='coerce')
            if target_numeric.isna().sum() > 0:
                print("   目标变量包含非数值，跳过统计检验")
                statistical_selected = variance_selected
            else:
                # 限制特征数量以避免内存问题
                max_features = min(100, len(variance_selected))
                test_cols = variance_selected[:max_features]
                
                # 确保样本数量一致
                common_index = df_selected.index.intersection(target_numeric.index)
                if len(common_index) < len(df_selected):
                    print(f"   样本数量不一致，使用共同索引: {len(common_index)} 个样本")
                    df_selected_aligned = df_selected.loc[common_index]
                    target_aligned = target_numeric.loc[common_index]
                else:
                    df_selected_aligned = df_selected
                    target_aligned = target_numeric
                
                selector = SelectKBest(score_func=f_classif, k=min(50, len(test_cols)))
                df_statistical = selector.fit_transform(df_selected_aligned[test_cols], target_aligned)
                statistical_selected = [test_cols[i] for i in selector.get_support(indices=True)]
                print(f"   统计检验法: {len(statistical_selected)} 个特征")
        except Exception as e:
            print(f"   统计检验法失败: {e}")
            statistical_selected = variance_selected
        
        # 3. 模型重要性选择
        print("4. 模型重要性选择...")
        try:
            # 限制特征数量
            max_features = min(50, len(statistical_selected))
            model_cols = statistical_selected[:max_features]
            
            # 确保样本数量一致
            common_index = df_selected.index.intersection(target.index)
            if len(common_index) < len(df_selected):
                print(f"   样本数量不一致，使用共同索引: {len(common_index)} 个样本")
                df_selected_aligned = df_selected.loc[common_index]
                target_aligned = target.loc[common_index]
            else:
                df_selected_aligned = df_selected
                target_aligned = target
            
            rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            rf.fit(df_selected_aligned[model_cols], target_aligned)
            
            # 获取特征重要性
            importances = rf.feature_importances_
            importance_df = pd.DataFrame({
                'feature': model_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # 选择重要性较高的特征 (限制数量)
            model_selected = importance_df.head(min(30, len(importance_df)))['feature'].tolist()
            print(f"   模型重要性选择: {len(model_selected)} 个特征")
        except Exception as e:
            print(f"   模型重要性选择失败: {e}")
            model_selected = statistical_selected
        
        # 4. 递归特征消除
        print("5. 递归特征消除...")
        try:
            # 进一步限制特征数量
            max_features = min(20, len(model_selected))
            rfe_cols = model_selected[:max_features]
            
            # 确保样本数量一致
            common_index = df_selected.index.intersection(target.index)
            if len(common_index) < len(df_selected):
                print(f"   样本数量不一致，使用共同索引: {len(common_index)} 个样本")
                df_selected_aligned = df_selected.loc[common_index]
                target_aligned = target.loc[common_index]
            else:
                df_selected_aligned = df_selected
                target_aligned = target
            
            estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
            rfe = RFE(estimator, n_features_to_select=min(15, len(rfe_cols)))
            rfe.fit(df_selected_aligned[rfe_cols], target_aligned)
            
            rfe_selected = [rfe_cols[i] for i in range(len(rfe_cols)) if rfe.support_[i]]
            print(f"   递归特征消除: {len(rfe_selected)} 个特征")
        except Exception as e:
            print(f"   递归特征消除失败: {e}")
            rfe_selected = model_selected
        
        # 最终选择
        final_selected = rfe_selected
        df_final = df_selected[final_selected]
        
        # 保存选择的特征列表用于transform阶段 - 确保在fitted_transformers中
        if not hasattr(self, 'fitted_transformers'):
            self.fitted_transformers = {}
        self.fitted_transformers['selected_features'] = final_selected
        
        print(f"\n特征选择完成: {df.shape[1]} -> {df_final.shape[1]} 个特征")
        return df_final
    
    def _correlation_based_selection(self, df: pd.DataFrame, target: pd.Series, 
                                   top_k: int = 100) -> List[str]:
        """基于相关性的特征选择"""
        correlations = {}
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                continue
            
            # 确保目标变量和特征长度一致
            if len(df[col]) != len(target):
                continue
                
            # 处理缺失值
            col_data = df[col].fillna(0)
            target_data = target.fillna(0)
            
            # 计算与目标变量的相关性
            try:
                corr = abs(np.corrcoef(col_data, target_data)[0, 1])
                if not np.isnan(corr):
                    correlations[col] = corr
            except:
                continue
        
        # 选择相关性最高的特征
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:top_k]]
        
        print(f"   选择 {len(selected_features)} 个高相关性特征")
        return selected_features
    
    def _mutual_information_selection(self, df: pd.DataFrame, target: pd.Series, 
                                    top_k: int = 100) -> List[str]:
        """互信息特征选择"""
        # 处理分类变量
        df_encoded = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
        
        # 确保数据长度一致
        if len(df_encoded) != len(target):
            # 找到共同索引
            common_idx = df_encoded.index.intersection(target.index)
            df_encoded = df_encoded.loc[common_idx]
            target = target.loc[common_idx]
        
        # 处理缺失值 - 互信息计算不能处理NaN
        df_encoded = df_encoded.fillna(0)
        
        # 计算互信息
        mi_scores = mutual_info_classif(df_encoded, target, random_state=self.random_state)
        mi_dict = dict(zip(df_encoded.columns, mi_scores))
        
        # 选择互信息最高的特征
        sorted_features = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:top_k]]
        
        print(f"   选择 {len(selected_features)} 个高互信息特征")
        return selected_features
    
    def _statistical_test_selection(self, df: pd.DataFrame, target: pd.Series, 
                                    p_value_threshold: float = 0.05) -> List[str]:
        """统计检验特征选择"""
        selected_features = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # 卡方检验用于分类变量
                try:
                    contingency_table = pd.crosstab(df[col], target)
                    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    if p_value < p_value_threshold:
                        selected_features.append(col)
                except:
                    pass
            else:
                # F检验用于数值变量
                try:
                    groups = [df[col][target == class_val] for class_val in target.unique()]
                    f_stat, p_value = stats.f_oneway(*groups)
                    if p_value < p_value_threshold:
                        selected_features.append(col)
                except:
                    pass
        
        print(f"   选择 {len(selected_features)} 个统计显著特征")
        return selected_features
    
    def _model_based_selection(self, df: pd.DataFrame, target: pd.Series, 
                             top_k: int = 100) -> List[str]:
        """基于模型的特征选择"""
        # 编码分类变量
        df_encoded = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
        
        # 确保数据长度一致
        if len(df_encoded) != len(target):
            # 找到共同索引
            common_idx = df_encoded.index.intersection(target.index)
            df_encoded = df_encoded.loc[common_idx]
            target = target.loc[common_idx]
        
        # 处理缺失值
        df_encoded = df_encoded.fillna(0)
        
        # 使用随机森林评估特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(df_encoded, target)
        
        # 获取特征重要性
        feature_importance = dict(zip(df_encoded.columns, rf.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:top_k]]
        
        print(f"   选择 {len(selected_features)} 个高重要性特征")
        return selected_features
    
    def _recursive_feature_elimination(self, df: pd.DataFrame, target: pd.Series, 
                                       n_features_to_select: int = 50) -> List[str]:
        """递归特征消除"""
        # 编码分类变量
        df_encoded = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
        
        # 确保数据长度一致
        if len(df_encoded) != len(target):
            # 找到共同索引
            common_idx = df_encoded.index.intersection(target.index)
            df_encoded = df_encoded.loc[common_idx]
            target = target.loc[common_idx]
        
        # 处理缺失值
        df_encoded = df_encoded.fillna(0)
        
        # 使用逻辑回归进行RFE
        estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(df_encoded, target)
        
        selected_features = df_encoded.columns[rfe.support_].tolist()
        
        print(f"   选择 {len(selected_features)} 个RFE特征")
        return selected_features
    
    def comprehensive_dimensionality_reduction(self, df: pd.DataFrame, 
                                             n_components: int = 50) -> pd.DataFrame:
        """
        全面降维
        """
        print("=== 全面降维阶段 ===")
        
        # 编码分类变量 - 确保只处理真正的分类列
        df_encoded = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
        
        # 确保所有列都是数值类型 - 严格检查每个列
        for col in df_encoded.columns:
            if not pd.api.types.is_numeric_dtype(df_encoded[col]):
                # 尝试转换为数值类型，如果失败则删除该列
                try:
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                    if df_encoded[col].isna().all():
                        df_encoded = df_encoded.drop(columns=[col])
                        print(f"   删除无法转换的列: {col}")
                except:
                    df_encoded = df_encoded.drop(columns=[col])
                    print(f"   删除无法转换的列: {col}")
        
        # 处理缺失值 - PCA不能处理NaN
        df_encoded = df_encoded.fillna(0)
        
        # 如果没有任何数值列，返回原数据
        if df_encoded.shape[1] == 0:
            print("   警告: 没有数值列可用于降维，返回原数据")
            return df
        
        df_reduced = df_encoded.copy()
        
        # 1. PCA降维
        print("1. PCA降维...")
        if df_encoded.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=self.random_state)
            pca_features = pca.fit_transform(df_encoded)
            
            # 添加PCA特征
            for i in range(min(n_components, pca_features.shape[1])):
                df_reduced[f'pca_{i+1}'] = pca_features[:, i]
            
            self.fitted_transformers['pca'] = pca
            print(f"   PCA解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
        
        # 2. t-SNE (用于可视化，不降维到特征中)
        print("2. t-SNE可视化...")
        if df_encoded.shape[1] > 10 and df_encoded.shape[0] < 10000:  # t-SNE计算成本高
            tsne = TSNE(n_components=2, random_state=self.random_state)
            tsne_features = tsne.fit_transform(df_encoded)
            
            # 保存t-SNE结果用于可视化
            self.fitted_transformers['tsne_features'] = tsne_features
        
        # 3. 特征聚类
        print("3. 特征聚类...")
        df_reduced = self._feature_clustering(df_reduced)
        
        print(f"   降维完成，最终特征数: {df_reduced.shape[1]}")
        return df_reduced
    
    def _feature_clustering(self, df: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
        """特征聚类"""
        df_clustered = df.copy()
        
        # 转置数据矩阵，对特征进行聚类 - 确保只使用数值列
        numeric_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        
        if len(numeric_cols) == 0:
            return df_clustered
        
        # 确保没有缺失值
        feature_matrix = df[numeric_cols].fillna(0).T
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # 为每个聚类创建代表性特征 (使用聚类中心)
        for cluster_id in range(n_clusters):
            cluster_features = [numeric_cols[i] for i in range(len(numeric_cols)) 
                              if cluster_labels[i] == cluster_id]
            if len(cluster_features) > 0:
                # 使用聚类内特征的平均值作为代表性特征 - 确保数值类型
                numeric_cluster_features = []
                for feature in cluster_features:
                    if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                        numeric_cluster_features.append(feature)
                
                if len(numeric_cluster_features) > 0:
                    df_clustered[f'cluster_{cluster_id}_mean'] = df[numeric_cluster_features].mean(axis=1)
                    df_clustered[f'cluster_{cluster_id}_std'] = df[numeric_cluster_features].std(axis=1)
        
        self.fitted_transformers['feature_kmeans'] = kmeans
        self.fitted_transformers['feature_clusters'] = cluster_labels
        
        return df_clustered
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换新数据 - 使用已拟合的转换器
        """
        print("=== 应用保存的特征工程转换 ===")
        
        df = X.copy()
        original_shape = df.shape
        
        # 1. 数据清洗
        print("1. 数据清洗...")
        df_clean = self._apply_data_cleaning_transforms(df)
        
        # 2. 特征转换
        print("2. 特征转换...")
        df_transformed = self._apply_feature_transformation_transforms(df_clean)
        
        # 3. 特征创建
        print("3. 特征创建...")
        df_created = self._apply_feature_creation_transforms(df_transformed)
        
        # 4. 特征选择 - 使用训练时保存的特征列表
        print("4. 特征选择...")
        df_selected = self._apply_feature_selection_transforms(df_created)
        
        # 5. 降维
        print("5. 降维...")
        df_final = self._apply_dimensionality_reduction_transforms(df_selected)
        
        print(f"转换完成: {original_shape} -> {df_final.shape}")
        return df_final
    
    def _apply_data_cleaning_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用数据清洗转换"""
        df_clean = df.copy()
        
        # 应用保存的缺失值处理转换器
        for col_name, transformer in self.fitted_transformers.items():
            if 'imputer' in col_name:
                col = col_name.replace('_knn_imputer', '')
                if col in df_clean.columns:
                    df_clean[col] = transformer.transform(df_clean[[col]])
        
        return df_clean
    
    def _apply_feature_transformation_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用特征转换"""
        df_transformed = df.copy()
        
        # 应用保存的数值变换
        for col_name, transformer in self.fitted_transformers.items():
            if 'quantile' in col_name:
                col = col_name.replace('_quantile', '')
                if col in df_transformed.columns:
                    transformed_values = transformer.transform(df_transformed[[col]])
                    df_transformed[col_name] = transformed_values.flatten()
            elif 'yeojohnson' in col_name:
                col = col_name.replace('_yeojohnson', '')
                if col in df_transformed.columns:
                    try:
                        from scipy.stats import yeojohnson
                        transformed_values, _ = yeojohnson(df_transformed[col])
                        df_transformed[col_name] = transformed_values
                    except:
                        pass
        
        return df_transformed
    
    def _apply_feature_creation_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用特征创建转换 - 重新创建交互和多项式特征"""
        df_created = df.copy()
        
        # 重新创建交互特征（如果存在相关转换器）
        numeric_cols = [col for col in df_created.columns if pd.api.types.is_numeric_dtype(df_created[col])]
        categorical_cols = [col for col in df_created.columns if not pd.api.types.is_numeric_dtype(df_created[col])]
        
        if len(numeric_cols) >= 2:
            print(f"   重新创建交互特征...")
            # 创建数值特征间的交互
            for i in range(min(len(numeric_cols), 5)):  # 限制数量避免过多
                for j in range(i+1, min(len(numeric_cols), 5)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    df_created[f'{col1}_x_{col2}'] = df_created[col1] * df_created[col2]
                    df_created[f'{col1}_+_{col2}'] = df_created[col1] + df_created[col2]
                    df_created[f'{col1}_-_{col2}'] = df_created[col1] - df_created[col2]
                    if df_created[col2].abs().max() > 0:  # 避免除零
                        df_created[f'{col1}_/_${col2}'] = df_created[col1] / df_created[col2]
        
        # 重新创建多项式特征（如果存在相关转换器）
        if len(numeric_cols) > 0:
            print(f"   重新创建多项式特征...")
            for col in numeric_cols[:3]:  # 限制数量
                df_created[f'{col}_squared'] = df_created[col] ** 2
                df_created[f'{col}_cubed'] = df_created[col] ** 3
                if df_created[col].abs().min() > 0:  # 避免除零
                    df_created[f'{col}_inverse'] = 1 / df_created[col]
        
        return df_created
    
    def _apply_feature_selection_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征选择转换 - 使用训练时保存的特征列表
        """
        print("  应用特征选择...")
        
        if 'selected_features' not in self.fitted_transformers:
            print("  警告: 未找到保存的特征选择列表，跳过特征选择")
            return df
        
        selected_features = self.fitted_transformers['selected_features']
        
        # 获取当前数据中的所有特征名称
        current_features = list(df.columns)
        
        # 找到同时存在于selected_features和当前数据中的特征
        available_features = []
        for feature in selected_features:
            if feature in current_features:
                available_features.append(feature)
        
        missing_features = [f for f in selected_features if f not in current_features]
        
        if missing_features:
            print(f"  警告: {len(missing_features)} 个特征不存在，将使用可用特征")
            if len(missing_features) <= 10:
                print(f"  缺失特征: {missing_features}")
            else:
                print(f"  缺失特征示例: {missing_features[:10]}")
        
        if available_features:
            df_selected = df[available_features]
            print(f"  特征选择: {df.shape[1]} -> {df_selected.shape[1]} 个特征")
            return df_selected
        else:
            print("  警告: 没有可用的选择特征，返回原始数据")
            return df
    
    def _apply_dimensionality_reduction_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用降维转换"""
        df_reduced = df.copy()
        
        # 应用PCA
        if 'pca' in self.fitted_transformers:
            pca = self.fitted_transformers['pca']
            numeric_cols = [col for col in df_reduced.columns if pd.api.types.is_numeric_dtype(df_reduced[col])]
            if len(numeric_cols) > 0:
                pca_features = pca.transform(df_reduced[numeric_cols])
                for i in range(pca_features.shape[1]):
                    df_reduced[f'pca_{i+1}'] = pca_features[:, i]
        
        return df_reduced

    def fit_transform(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """
        完整的特征工程流程
        """
        self.target = target
        
        print("开始全面特征工程处理...")
        start_time = datetime.now()
        
        # 1. 数据清洗
        df_clean = self.comprehensive_data_cleaning(df)
        
        # 2. 特征转换
        df_transformed = self.comprehensive_feature_transformation(df_clean)
        
        # 3. 特征创建
        df_created = self.comprehensive_feature_creation(df_transformed)
        
        # 4. 特征选择 (如果有目标变量)
        if target is not None:
            df_selected = self.comprehensive_feature_selection(df_created, target)
        else:
            df_selected = df_created
        
        # 5. 降维
        df_reduced = self.comprehensive_dimensionality_reduction(df_selected)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n全面特征工程处理完成!")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"特征变化: {df.shape[1]} -> {df_reduced.shape[1]}")
        print(f"特征增长率: {(df_reduced.shape[1] / df.shape[1] - 1) * 100:.2f}%")
        
        return df_reduced
    
    def save_transformers(self, filepath: str):
        """保存转换器"""
        joblib.dump(self.fitted_transformers, filepath)
        print(f"转换器已保存到: {filepath}")
        
    def load_transformers(self, filepath: str):
        """加载转换器"""
        self.fitted_transformers = joblib.load(filepath)
        print(f"转换器已从 {filepath} 加载")


# 使用示例和测试函数
def test_comprehensive_feature_engineering():
    """测试全面特征工程系统 - 使用真实数据集"""
    print("=== 测试全面特征工程系统 ===")
    
    # 加载真实数据集
    try:
        train_data = pd.read_csv('/Users/qingguo/Documents/project/carPricePredict/data/train.csv')
        print(f"成功加载训练数据，形状: {train_data.shape}")
        print(f"训练数据列名: {list(train_data.columns)}")
    except Exception as e:
        print(f"加载训练数据失败: {e}")
        return None, None, None
    
    # 分离特征和目标变量
    target_col = '是否违约'  # 根据数据集确定目标变量列名
    if target_col not in train_data.columns:
        print(f"警告: 目标列 '{target_col}' 不存在，尝试其他可能的目标列名")
        # 尝试其他可能的目标列名
        possible_targets = ['是否违约', 'is_default', 'default', 'target', 'y']
        target_col = None
        for col in possible_targets:
            if col in train_data.columns:
                target_col = col
                break
        
        if target_col is None:
            print("错误: 无法找到目标变量列")
            return None, None, None
    
    print(f"使用目标列: {target_col}")
    
    # 分离特征和目标
    X = train_data.drop(columns=[target_col])
    y = train_data[target_col]
    
    print(f"特征数据形状: {X.shape}")
    print(f"目标数据形状: {y.shape}")
    print(f"目标变量分布: {y.value_counts().to_dict()}")
    
    # 应用全面特征工程
    fe_system = ComprehensiveFeatureEngineering(random_state=42)
    X_transformed = fe_system.fit_transform(X, y)
    
    print(f"\n变换后数据形状: {X_transformed.shape}")
    print(f"特征工程完成！特征数量从 {X.shape[1]} 增加到 {X_transformed.shape[1]}")
    
    # 保存转换器
    fe_system.save_transformers('/Users/qingguo/Documents/project/carPricePredict/comprehensive_fe_transformers.pkl')
    
    return X_transformed, y, fe_system


if __name__ == "__main__":
    # 运行测试
    X_transformed, y, fe_system = test_comprehensive_feature_engineering()
    
    print("\n=== 全面特征工程系统测试完成 ===")
    print(f"最终特征数量: {X_transformed.shape[1]}")
    print(f"特征工程效果评估完成！")