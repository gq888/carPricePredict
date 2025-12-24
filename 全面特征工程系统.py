import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import mutual_info_score
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import warnings
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import joblib

# 忽略警告
warnings.filterwarnings('ignore')

class ComprehensiveFeatureEngineering:
    """
    全面特征工程系统
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.fitted_transformers = {}
        self.target = None
        
    def comprehensive_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全面数据清洗
        """
        print("=== 数据清洗阶段 ===")
        
        original_shape = df.shape
        print(f"原始数据形状: {original_shape}")
        
        # 1. 基础信息统计
        print("1. 基础信息统计...")
        print(f"   数据形状: {df.shape}")
        print(f"   缺失值比例: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")
        print(f"   重复行数: {df.duplicated().sum()}")
        
        # 2. 处理重复数据
        print("2. 处理重复数据...")
        df_clean = df.drop_duplicates()
        print(f"   删除重复行: {original_shape[0] - df_clean.shape[0]}")
        
        # 3. 特征类型识别与优化
        print("3. 特征类型识别与优化...")
        df_clean = self._identify_feature_types(df_clean)
        
        # 4. 缺失值处理
        print("4. 缺失值处理...")
        df_clean = self._advanced_missing_value_imputation(df_clean)
        
        # 5. 异常值检测与处理
        print("5. 异常值检测与处理...")
        df_clean = self._comprehensive_outlier_detection(df_clean)
        
        # 6. 数据类型优化
        print("6. 数据类型优化...")
        df_clean = self._optimize_data_types(df_clean)
        
        # 保存清洗后的索引，用于同步目标变量
        self.fitted_transformers['index_after_cleaning'] = df_clean.index
        
        print(f"数据清洗完成: {original_shape} -> {df_clean.shape}")
        return df_clean
    
    def _identify_feature_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别特征类型"""
        df_identified = df.copy()
        
        # 自动识别数值特征
        numeric_cols = df_identified.select_dtypes(include=[np.number]).columns.tolist()
        
        # 自动识别分类特征
        categorical_cols = df_identified.select_dtypes(include=['object']).columns.tolist()
        
        # 识别时间特征
        datetime_cols = []
        for col in df_identified.columns:
            if '时间' in col or '日期' in col or 'date' in col.lower() or 'time' in col.lower():
                try:
                    df_identified[col] = pd.to_datetime(df_identified[col])
                    datetime_cols.append(col)
                except:
                    pass
        
        print(f"   数值特征: {len(numeric_cols)} 个")
        print(f"   分类特征: {len(categorical_cols)} 个")
        print(f"   时间特征: {len(datetime_cols)} 个")
        
        return df_identified
    
    def _advanced_missing_value_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级缺失值处理"""
        df_imputed = df.copy()
        
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
                
            missing_rate = df[col].isnull().sum() / len(df)
            print(f"   处理 {col} (缺失率: {missing_rate:.2%})")
            
            if missing_rate > 0.5:  # 缺失率超过50%
                print(f"     缺失率过高，创建缺失指示器")
                df_imputed[f'{col}_was_missing'] = df[col].isnull().astype(int)
                
            if df[col].dtype in ['object', 'category']:
                # 分类变量使用众数填充
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df_imputed[col] = df_imputed[col].fillna(mode_value[0])
            else:
                # 数值变量使用KNN填充
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                    self.fitted_transformers[f'{col}_knn_imputer'] = imputer
                except:
                    # 如果KNN失败，使用中位数
                    median_value = df[col].median()
                    df_imputed[col] = df_imputed[col].fillna(median_value)
        
        return df_imputed
    
    def _comprehensive_outlier_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """全面异常值检测"""
        df_outliers = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                continue
                
            # IQR方法
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Z-score方法
            z_scores = np.abs(stats.zscore(df[col]))
            
            # 修正Z-score方法 (使用中位数)
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            modified_z_scores = 0.6745 * (df[col] - median) / mad
            
            # 孤立森林
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
                outlier_labels = iso_forest.fit_predict(df[[col]])
            except:
                outlier_labels = np.zeros(len(df))  # 如果没有异常值，全部标记为正常
            
            # 综合判断异常值
            outlier_mask = (
                (df[col] < lower_bound) | (df[col] > upper_bound) |
                (z_scores > 3) |
                (np.abs(modified_z_scores) > 3.5) |
                (outlier_labels == -1)
            )
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                print(f"   {col}: 检测到 {outlier_count} 个异常值 ({outlier_count/len(df):.2%})")
                
                # 创建异常值指示器
                df_outliers[f'{col}_is_outlier'] = outlier_mask.astype(int)
                
                # 使用边界值替换异常值
                df_outliers.loc[outlier_mask, col] = np.where(
                    df_outliers.loc[outlier_mask, col] < lower_bound,
                    lower_bound,
                    upper_bound
                )
        
        return df_outliers
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型"""
        df_optimized = df.copy()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object' and 'datetime' not in str(col_type):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)
        
        return df_optimized
    
    def comprehensive_feature_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全面特征转换
        """
        print("=== 特征转换阶段 ===")
        
        df_transformed = df.copy()
        
        # 1. 数值特征标准化
        print("1. 数值特征标准化...")
        df_transformed = self._advanced_numerical_scaling(df_transformed)
        
        # 2. 分类特征编码
        print("2. 分类特征编码...")
        df_transformed = self._advanced_categorical_encoding(df_transformed)
        
        # 3. 时间特征处理
        print("3. 时间特征处理...")
        df_transformed = self._temporal_feature_engineering(df_transformed)
        
        # 4. 数值变换
        print("4. 数值变换...")
        df_transformed = self._advanced_numerical_transformation(df_transformed)
        
        # 5. 数据类型验证
        print("5. 数据类型验证...")
        df_transformed = self._validate_data_types(df_transformed)
        
        print(f"特征转换完成，最终特征数: {df_transformed.shape[1]}")
        return df_transformed
    
    def _advanced_numerical_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级数值特征标准化"""
        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                continue
                
            # 检查数据分布
            skewness = df[col].skew()
            
            if abs(skewness) > 1:  # 高偏度数据使用RobustScaler
                scaler = RobustScaler()
                scaled_values = scaler.fit_transform(df[[col]])
                df_scaled[f'{col}_robust_scaled'] = scaled_values.flatten()
                self.fitted_transformers[f'{col}_robust_scaler'] = scaler
            else:  # 正态分布数据使用StandardScaler
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df[[col]])
                df_scaled[f'{col}_standard_scaled'] = scaled_values.flatten()
                self.fitted_transformers[f'{col}_standard_scaler'] = scaler
        
        return df_scaled
    
    def _advanced_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级分类特征编码"""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].nunique() <= 2:  # 二分类使用标签编码
                le = LabelEncoder()
                df_encoded[f'{col}_label_encoded'] = le.fit_transform(df[col].astype(str))
                self.fitted_transformers[f'{col}_label_encoder'] = le
            elif df[col].nunique() <= 10:  # 少量类别使用独热编码
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
            else:  # 高基数类别使用频率编码
                freq_map = df[col].value_counts().to_dict()
                df_encoded[f'{col}_freq_encoded'] = df[col].map(freq_map)
        
        return df_encoded
    
    def _temporal_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间特征工程"""
        df_temporal = df.copy()
        
        datetime_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
            except:
                pass
        
        for col in datetime_cols:
            # 基础时间特征
            df_temporal[f'{col}_year'] = df_temporal[col].dt.year
            df_temporal[f'{col}_month'] = df_temporal[col].dt.month
            df_temporal[f'{col}_day'] = df_temporal[col].dt.day
            df_temporal[f'{col}_weekday'] = df_temporal[col].dt.weekday
            df_temporal[f'{col}_quarter'] = df_temporal[col].dt.quarter
            
            # 高级时间特征
            df_temporal[f'{col}_is_weekend'] = (df_temporal[col].dt.weekday >= 5).astype(int)
            df_temporal[f'{col}_day_of_year'] = df_temporal[col].dt.dayofyear
            df_temporal[f'{col}_week_of_year'] = df_temporal[col].dt.isocalendar().week
        
        return df_temporal
    
    def _advanced_numerical_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级数值变换"""
        df_transformed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0 or (df[col] == 0).sum() > 0:
                continue
                
            # 对数变换 (适用于右偏数据)
            if df[col].min() > 0 and df[col].skew() > 1:
                try:
                    log_transformed = np.log(df[col])
                    if not np.isinf(log_transformed).any() and not np.isnan(log_transformed).any():
                        df_transformed[f'{col}_log'] = log_transformed
                except:
                    pass
            
            # 平方根变换 (适用于中度偏斜数据)
            if df[col].min() >= 0:
                try:
                    sqrt_transformed = np.sqrt(df[col])
                    if not np.isinf(sqrt_transformed).any() and not np.isnan(sqrt_transformed).any():
                        df_transformed[f'{col}_sqrt'] = sqrt_transformed
                except:
                    pass
            
            # Box-Cox变换 (需要正数)
            if df[col].min() > 0 and abs(df[col].skew()) > 0.5:
                try:
                    boxcox_transformed, _ = boxcox(df[col])
                    if not np.isinf(boxcox_transformed).any() and not np.isnan(boxcox_transformed).any():
                        df_transformed[f'{col}_boxcox'] = boxcox_transformed
                        self.fitted_transformers[f'{col}_boxcox_lambda'] = _
                except:
                    pass
            
            # Yeo-Johnson变换 (允许负数)
            if abs(df[col].skew()) > 0.5:
                try:
                    yj_transformed, _ = yeojohnson(df[col])
                    if not np.isinf(yj_transformed).any() and not np.isnan(yj_transformed).any():
                        df_transformed[f'{col}_yeojohnson'] = yj_transformed
                        self.fitted_transformers[f'{col}_yeojohnson_lambda'] = _
                except:
                    pass
            
            # 分位数变换 (转换为均匀分布)
            try:
                from sklearn.preprocessing import QuantileTransformer
                qt = QuantileTransformer(output_distribution='uniform')
                quantile_transformed = qt.fit_transform(df[[col]])
                df_transformed[f'{col}_quantile'] = quantile_transformed.flatten()
                self.fitted_transformers[f'{col}_quantile_transformer'] = qt
            except:
                pass
            
            # 幂变换
            for power in [2, 3]:
                try:
                    power_transformed = df[col] ** power
                    if not np.isinf(power_transformed).any() and not np.isnan(power_transformed).any():
                        df_transformed[f'{col}_power_{power}'] = power_transformed
                except:
                    pass
            
            # 倒数变换
            if df[col].min() > 0:
                try:
                    reciprocal_transformed = 1 / df[col]
                    if not np.isinf(reciprocal_transformed).any() and not np.isnan(reciprocal_transformed).any():
                        df_transformed[f'{col}_reciprocal'] = reciprocal_transformed
                except:
                    pass
            
            # 特征分箱
            if df[col].nunique() > 10:
                try:
                    # 等宽分箱
                    equal_width_bins = pd.cut(df[col], bins=5, labels=False)
                    df_transformed[f'{col}_equal_width_binned'] = equal_width_bins
                    
                    # 等频分箱
                    equal_freq_bins = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
                    df_transformed[f'{col}_equal_freq_binned'] = equal_freq_bins
                except:
                    pass
        
        return df_transformed
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据类型验证"""
        df_validated = df.copy()
        
        # 识别所有非数值列
        non_numeric_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        
        print(f"   需要验证的非数值列: {len(non_numeric_cols)} 个")
        
        # 尝试将非数值列转换为数值类型
        for col in non_numeric_cols:
            try:
                # 首先尝试直接转换为数值类型
                original_data = df_validated[col].copy()
                df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
                
                # 计算转换成功率
                non_null_ratio = df_validated[col].notna().sum() / len(df_validated[col])
                
                if non_null_ratio > 0.7:  # 转换成功率超过70%
                    print(f"   转换列 {col} 为数值类型 (成功率: {non_null_ratio:.1%})")
                    
                    # 如果仍有缺失值，用中位数填充
                    if df_validated[col].isnull().sum() > 0:
                        median_value = df_validated[col].median()
                        df_validated[col] = df_validated[col].fillna(median_value)
                        print(f"     用中位数 {median_value} 填充 {df_validated[col].isnull().sum()} 个缺失值")
                else:
                    # 转换成功率太低，回滚到原始数据
                    print(f"   列 {col} 转换成功率太低 ({non_null_ratio:.1%})，保留原始类型")
                    df_validated[col] = original_data
                    
                    # 对于原始分类列，尝试标签编码
                    if df[col].dtype == 'object':
                        try:
                            le = LabelEncoder()
                            df_validated[f'{col}_label_encoded'] = le.fit_transform(df[col].astype(str))
                            self.fitted_transformers[f'{col}_label_encoder'] = le
                            print(f"     创建标签编码特征: {col}_label_encoded")
                        except:
                            pass
                            
            except Exception as e:
                print(f"   列 {col} 转换失败: {e}")
                continue
        
        # 处理数值列中的无穷值
        numeric_cols = df_validated.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_validated[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                # 检查无穷值
                inf_mask = np.isinf(df_validated[col])
                if inf_mask.any():
                    print(f"   处理列 {col} 中的无穷值 ({inf_mask.sum()} 个)")
                    # 将无穷值替换为NaN，然后用中位数填充
                    df_validated.loc[inf_mask, col] = np.nan
                    median_value = df_validated[col].median()
                    if pd.notna(median_value):
                        df_validated[col] = df_validated[col].fillna(median_value)
                    else:
                        # 如果中位数也是NaN，用0填充
                        df_validated[col] = df_validated[col].fillna(0)
        
        # 删除转换后全为NaN的列
        cols_to_drop = []
        for col in df_validated.columns:
            if df_validated[col].isnull().all():
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df_validated = df_validated.drop(columns=cols_to_drop)
            print(f"   删除全为NaN的列: {cols_to_drop}")
        
        return df_validated
    
    def comprehensive_feature_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全面特征创建
        """
        print("=== 特征创建阶段 ===")
        
        df_created = df.copy()
        
        # 1. 特征交互
        print("1. 特征交互...")
        df_created = self._feature_interaction_creation(df_created)
        
        # 2. 多项式特征
        print("2. 多项式特征...")
        df_created = self._polynomial_feature_creation(df_created)
        
        # 3. 聚合特征
        print("3. 聚合特征...")
        df_created = self._aggregation_feature_creation(df_created)
        
        # 4. 领域特定特征
        print("4. 领域特定特征...")
        df_created = self._domain_specific_features(df_created)
        
        print(f"特征创建完成，最终特征数: {df_created.shape[1]}")
        return df_created
    
    def _feature_interaction_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征交互创建"""
        df_interactions = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 数值特征间的交互
        for i in range(min(len(numeric_cols), 10)):  # 限制数量避免组合爆炸
            for j in range(i+1, min(len(numeric_cols), 10)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                
                # 创建交互特征
                df_interactions[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df_interactions[f'{col1}_+_{col2}'] = df[col1] + df[col2]
                df_interactions[f'{col1}_-_{col2}'] = df[col1] - df[col2]
                
                # 避免除零
                if df[col2].abs().min() > 0:
                    df_interactions[f'{col1}_/_${col2}'] = df[col1] / df[col2]
        
        return df_interactions
    
    def _polynomial_feature_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        """多项式特征创建"""
        df_polynomial = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:5]:  # 限制数量
            if df[col].isnull().sum() == 0 and df[col].dtype != 'object':
                # 平方和立方
                df_polynomial[f'{col}_squared'] = df[col] ** 2
                df_polynomial[f'{col}_cubed'] = df[col] ** 3
                
                # 平方根 (非负数)
                if df[col].min() >= 0:
                    sqrt_values = np.sqrt(df[col])
                    if not np.isnan(sqrt_values).any() and not np.isinf(sqrt_values).any():
                        df_polynomial[f'{col}_sqrt'] = sqrt_values
                
                # 倒数 (非零)
                if df[col].abs().min() > 0:
                    reciprocal_values = 1 / df[col]
                    if not np.isnan(reciprocal_values).any() and not np.isinf(reciprocal_values).any():
                        df_polynomial[f'{col}_inverse'] = reciprocal_values
        
        return df_polynomial
    
    def _aggregation_feature_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        """聚合特征创建"""
        df_aggregated = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            # 统计聚合特征
            df_aggregated['numeric_sum'] = df[numeric_cols].sum(axis=1)
            df_aggregated['numeric_mean'] = df[numeric_cols].mean(axis=1)
            df_aggregated['numeric_std'] = df[numeric_cols].std(axis=1)
            df_aggregated['numeric_min'] = df[numeric_cols].min(axis=1)
            df_aggregated['numeric_max'] = df[numeric_cols].max(axis=1)
            df_aggregated['numeric_range'] = df_aggregated['numeric_max'] - df_aggregated['numeric_min']
            
            # 比例特征
            for col in numeric_cols[:3]:
                df_aggregated[f'{col}_ratio_sum'] = df[col] / (df_aggregated['numeric_sum'] + 1e-8)
                df_aggregated[f'{col}_ratio_mean'] = df[col] / (df_aggregated['numeric_mean'] + 1e-8)
        
        return df_aggregated
    
    def _domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """领域特定特征"""
        df_domain = df.copy()
        
        # 金融领域特征
        if '收入' in df.columns and '支出' in df.columns:
            df_domain['收支比'] = df['收入'] / (df['支出'] + 1e-8)
            df_domain['净收入'] = df['收入'] - df['支出']
        
        if '贷款金额' in df.columns and '收入' in df.columns:
            df_domain['贷款收入比'] = df['贷款金额'] / (df['收入'] + 1e-8)
        
        # 年龄相关特征
        if '年龄' in df.columns:
            df_domain['年龄段'] = pd.cut(df['年龄'], bins=[0, 25, 35, 45, 55, 100], 
                                         labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            df_domain['是否青年'] = (df['年龄'] <= 35).astype(int)
            df_domain['是否中年'] = ((df['年龄'] > 35) & (df['年龄'] <= 55)).astype(int)
            df_domain['是否老年'] = (df['年龄'] > 55).astype(int)
        
        # 工作年限相关特征
        if '工作年限' in df.columns:
            # 确保工作年限是数值类型
            if pd.api.types.is_numeric_dtype(df['工作年限']):
                df_domain['工作经验等级'] = pd.cut(df['工作年限'], bins=[-1, 0, 2, 5, 10, 50], 
                                                   labels=['无经验', '新手', '初级', '中级', '高级'])
                df_domain['是否新手'] = (df['工作年限'] <= 2).astype(int)
                df_domain['是否资深'] = (df['工作年限'] > 5).astype(int)
            else:
                print(f"   警告: 工作年限列不是数值类型，跳过工作年限特征创建")
                # 尝试创建标签编码
                try:
                    le = LabelEncoder()
                    df_domain['工作年限_encoded'] = le.fit_transform(df['工作年限'].astype(str))
                    self.fitted_transformers['工作年限_label_encoder'] = le
                except:
                    pass
        
        return df_domain
    
    def comprehensive_feature_selection(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        全面特征选择
        """
        print("=== 特征选择阶段 ===")
        
        # 1. 移除低方差特征
        print("1. 移除低方差特征...")
        df_selected = self._remove_low_variance_features(df)
        
        # 2. 基于统计检验的特征选择
        print("2. 基于统计检验的特征选择...")
        statistical_selected = self._statistical_based_selection(df_selected, target)
        
        # 3. 基于模型重要性的特征选择
        print("3. 基于模型重要性的特征选择...")
        model_selected = self._model_based_selection(df_selected[statistical_selected], target)
        
        # 4. 递归特征消除
        print("4. 递归特征消除...")
        final_selected = self._recursive_feature_elimination(df_selected[model_selected], target)
        
        # 保存最终选择的特征列表
        self.fitted_transformers['selected_features'] = final_selected
        
        df_final = df_selected[final_selected]
        
        print(f"特征选择完成: {df.shape[1]} -> {df_final.shape[1]} 个特征")
        return df_final
    
    def _remove_low_variance_features(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """移除低方差特征"""
        # 计算每个特征的方差
        variances = df.var(numeric_only=True)
        
        # 找到低方差特征
        low_variance_cols = variances[variances < threshold].index.tolist()
        
        # 保留方差较高的特征
        high_variance_cols = [col for col in df.columns if col not in low_variance_cols]
        
        print(f"   移除 {len(low_variance_cols)} 个低方差特征")
        return df[high_variance_cols]
    
    def _statistical_based_selection(self, df: pd.DataFrame, target: pd.Series) -> List[str]:
        """基于统计检验的特征选择"""
        selected_features = []
        
        for col in df.columns:
            try:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    # 卡方检验用于分类变量
                    contingency_table = pd.crosstab(df[col], target)
                    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    if p_value < 0.05:
                        selected_features.append(col)
                else:
                    # F检验用于数值变量
                    groups = [df[col][target == class_val] for class_val in target.unique()]
                    f_stat, p_value = stats.f_oneway(*groups)
                    if p_value < 0.05:
                        selected_features.append(col)
            except Exception as e:
                print(f"   特征 {col} 统计检验失败: {e}")
                continue
        
        print(f"   统计检验选择 {len(selected_features)} 个特征")
        return selected_features
    
    def _model_based_selection(self, df: pd.DataFrame, target: pd.Series) -> List[str]:
        """基于模型重要性的特征选择"""
        # 编码分类变量
        df_encoded = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
        
        # 处理缺失值
        df_encoded = df_encoded.fillna(0)
        
        # 使用随机森林评估特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(df_encoded, target)
        
        # 获取特征重要性
        feature_importance = dict(zip(df_encoded.columns, rf.feature_importances_))
        
        # 选择重要性前50%的特征
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        n_select = max(1, len(sorted_features) // 2)
        selected_features = [feature for feature, _ in sorted_features[:n_select]]
        
        print(f"   模型重要性选择 {len(selected_features)} 个特征")
        return selected_features
    
    def _recursive_feature_elimination(self, df: pd.DataFrame, target: pd.Series) -> List[str]:
        """递归特征消除"""
        # 编码分类变量
        df_encoded = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
        
        # 处理缺失值
        df_encoded = df_encoded.fillna(0)
        
        # 使用逻辑回归进行RFE
        estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # 选择特征数量 (至少保留10个，最多保留50个)
        n_features_to_select = min(50, max(10, df_encoded.shape[1] // 3))
        
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(df_encoded, target)
        
        selected_features = df_encoded.columns[rfe.support_].tolist()
        
        print(f"   递归特征消除选择 {len(selected_features)} 个特征")
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
        
        # 处理无穷值 - 这是关键修复
        for col in df_encoded.columns:
            if df_encoded[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # 检查无穷值
                inf_mask = np.isinf(df_encoded[col])
                if inf_mask.any():
                    print(f"   处理列 {col} 中的无穷值: {inf_mask.sum()} 个")
                    # 将无穷值替换为0（或列的中位数）
                    df_encoded[col] = df_encoded[col].replace([np.inf, -np.inf], 0)
        
        # 再次检查无穷值
        for col in df_encoded.columns:
            if df_encoded[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # 检查是否还有无穷值
                if np.isinf(df_encoded[col]).any():
                    print(f"   警告: 列 {col} 仍存在无穷值，将删除该列")
                    df_encoded = df_encoded.drop(columns=[col])
        
        # 如果没有任何数值列，返回原数据
        if df_encoded.shape[1] == 0:
            print("   警告: 没有数值列可用于降维，返回原数据")
            return df
        
        df_reduced = df_encoded.copy()
        
        # 1. PCA降维
        print("1. PCA降维...")
        if df_encoded.shape[1] > n_components:
            try:
                pca = PCA(n_components=n_components, random_state=self.random_state)
                pca_features = pca.fit_transform(df_encoded)
                
                # 添加PCA特征
                for i in range(min(n_components, pca_features.shape[1])):
                    df_reduced[f'pca_{i+1}'] = pca_features[:, i]
                
                self.fitted_transformers['pca'] = pca
                print(f"   PCA解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
            except Exception as e:
                print(f"   PCA降维失败: {e}")
                # 如果PCA失败，跳过降维
                pass
        
        # 2. t-SNE (用于可视化，不降维到特征中)
        print("2. t-SNE可视化...")
        if df_encoded.shape[1] > 10 and df_encoded.shape[0] < 10000:  # t-SNE计算成本高
            try:
                tsne = TSNE(n_components=2, random_state=self.random_state)
                tsne_features = tsne.fit_transform(df_encoded)
                
                # 保存t-SNE结果用于可视化
                self.fitted_transformers['tsne_features'] = tsne_features
            except Exception as e:
                print(f"   t-SNE可视化失败: {e}")
                pass
        
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
        
        # 确保没有缺失值和无穷值
        feature_matrix = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0).T
        
        # K-means聚类
        try:
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
            
        except Exception as e:
            print(f"   特征聚类失败: {e}")
        
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
                # 确保没有无穷值
                pca_data = df_reduced[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
                pca_features = pca.transform(pca_data)
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
        
        # 如果数据清洗删除了行，需要同步更新目标变量
        if target is not None and 'index_after_cleaning' in self.fitted_transformers:
            cleaned_index = self.fitted_transformers['index_after_cleaning']
            if len(cleaned_index) != len(target):
                print(f"数据清洗删除了 {len(target) - len(cleaned_index)} 行，同步更新目标变量")
                target = target.loc[cleaned_index]
        
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