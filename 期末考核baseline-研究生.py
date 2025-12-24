#!/usr/bin/env python
# coding: utf-8

"""
增强版汽车价格预测基线脚本
实现了四个核心阶段的模块化封装，包含缓存机制和依赖检查
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文字体（macOS系统）
# 优先使用系统自带的中文字体
def setup_chinese_font():
    """设置中文字体配置"""
    chinese_fonts = ['Heiti TC', 'STHeiti', 'Arial Unicode MS', 'LiHei Pro', 'LiSong Pro']
    
    for font_name in chinese_fonts:
        try:
            # 检查字体是否可用
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            if font_name in available_fonts:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用中文字体: {font_name}")
                return True
        except Exception as e:
            continue
    
    print("⚠ 未找到合适的中文字体，图表中的中文可能显示为方框")
    print("  建议安装中文字体或使用英文字体替代")
    return False

# 初始化字体配置
setup_chinese_font()

# 全局数据存储
global_data = {}

from cache_util import load_cache, save_cache, check_dependencies

def data_cleaning_stage():
    """数据清洗阶段 - 闭包函数"""
    def clean_data():
        print("=== 数据清洗阶段 ===")
        
        # 检查依赖项
        dependencies = {
            'train_raw': global_data.get('train_raw'),
            'test_raw': global_data.get('test_raw')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'data_cleaning'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取原始数据
        train = global_data['train_raw'].copy()
        test = global_data['test_raw'].copy()
        
        print(f"原始数据形状 - 训练集: {train.shape}, 测试集: {test.shape}")
        
        # 1. 处理缺失值
        print("1. 处理缺失值...")
        # 数值列用中位数填充
        numeric_cols = train.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in train.columns:
                train[col].fillna(train[col].median(), inplace=True)
                if col in test.columns:
                    test[col].fillna(train[col].median(), inplace=True)
        
        # 分类列用众数填充
        categorical_cols = train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in train.columns:
                mode_val = train[col].mode()[0] if not train[col].mode().empty else 'Unknown'
                train[col].fillna(mode_val, inplace=True)
                if col in test.columns:
                    test[col].fillna(mode_val, inplace=True)
        
        # 2. 处理异常值
        print("2. 处理异常值...")
        # 移除明显异常的行
        train = train[train['贷款总额'] > 0]
        test = test[test['贷款总额'] > 0]
        
        # 3. 数据类型转换
        print("3. 数据类型转换...")
        # 将日期字符串转换为数值特征
        date_cols = ['最早信用账户开通月']
        for col in date_cols:
            if col in train.columns:
                # 简化的日期处理，转换为年份差值
                train[col] = pd.to_datetime(train[col], errors='coerce')
                test[col] = pd.to_datetime(test[col], errors='coerce')
                
                # 计算距离现在的年份差
                reference_date = pd.to_datetime('2020-01-01')
                train[f'{col}_years'] = (reference_date - train[col]).dt.days / 365.25
                test[f'{col}_years'] = (reference_date - test[col]).dt.days / 365.25
                
                # 填充缺失值
                train[f'{col}_years'].fillna(train[f'{col}_years'].median(), inplace=True)
                test[f'{col}_years'].fillna(train[f'{col}_years'].median(), inplace=True)
                
                # 删除原始日期列
                train.drop(columns=[col], inplace=True)
                test.drop(columns=[col], inplace=True)
        
        print(f"清洗后数据形状 - 训练集: {train.shape}, 测试集: {test.shape}")
        
        result = {'train_clean': train, 'test_clean': test}
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return clean_data

def eda_stage():
    """探索性数据分析阶段 - 闭包函数"""
    def perform_eda():
        print("=== 探索性数据分析阶段 ===")
        
        # 检查依赖项
        dependencies = {
            'train_clean': global_data.get('train_clean'),
            'test_clean': global_data.get('test_clean')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'eda'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取清洗后的数据
        train = global_data['train_clean'].copy()
        test = global_data['test_clean'].copy()
        
        print(f"EDA分析 - 训练集: {train.shape}, 测试集: {test.shape}")
        
        # 1. 数据概览
        print("1. 数据概览...")
        print("训练集信息:")
        print(train.info())
        print("\n测试集信息:")
        print(test.info())
        
        # 2. 数值特征分析
        print("2. 数值特征分析...")
        numeric_cols = train.select_dtypes(include=[np.number]).columns
        
        # 创建综合可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('数据探索性分析', fontsize=16)
        
        # 训练集标签分布
        if '是否违约' in train.columns:
            train['是否违约'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
            axes[0,0].set_title('训练集标签分布')
            axes[0,0].set_xlabel('是否违约')
            axes[0,0].set_ylabel('数量')
            axes[0,0].tick_params(axis='x', rotation=0)
        
        # 贷款总额分布
        if '贷款总额' in train.columns:
            train['贷款总额'].hist(bins=50, ax=axes[0,1], color='lightblue', alpha=0.7)
            axes[0,1].set_title('贷款总额分布')
            axes[0,1].set_xlabel('贷款总额')
            axes[0,1].set_ylabel('频次')
        
        # 相关性热力图（前10个数值特征）
        if len(numeric_cols) > 0:
            corr_cols = numeric_cols[:min(10, len(numeric_cols))]
            if '是否违约' in train.columns:
                corr_cols = list(corr_cols) + ['是否违约'] if '是否违约' not in corr_cols else corr_cols
            
            corr_matrix = train[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
            axes[1,0].set_title('特征相关性热力图')
        
        # 缺失值分析
        missing_data = train.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            missing_data.plot(kind='bar', ax=axes[1,1], color='orange')
            axes[1,1].set_title('缺失值分析')
            axes[1,1].set_xlabel('特征')
            axes[1,1].set_ylabel('缺失值数量')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, '无缺失值', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('缺失值分析')
        
        plt.tight_layout()
        plt.savefig('/Users/qingguo/Documents/project/carPricePredict/eda_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 特征统计
        print("3. 特征统计...")
        print("数值特征描述性统计:")
        print(train[numeric_cols].describe())
        
        print("\n分类特征统计:")
        categorical_cols = train.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # 只显示前5个分类特征
            print(f"\n{col}:")
            print(train[col].value_counts().head())
        
        result = {'eda_completed': True}
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return perform_eda

def feature_engineering_stage():
    """特征工程阶段 - 闭包函数"""
    def engineer_features():
        print("=== 特征工程阶段 ===")
        
        # 检查依赖项
        dependencies = {
            'train_clean': global_data.get('train_clean'),
            'test_clean': global_data.get('test_clean')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'feature_engineering'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取清洗后的数据
        train = global_data['train_clean'].copy()
        test = global_data['test_clean'].copy()
        
        print(f"特征工程 - 训练集: {train.shape}, 测试集: {test.shape}")
        
        # 尝试使用全面特征工程系统
        try:
            from 全面特征工程系统 import ComprehensiveFeatureEngineering
            
            print("使用全面特征工程系统...")
            
            # 分离训练集的特征和目标
            if '是否违约' in train.columns:
                X_train = train.drop(columns=['是否违约'])
                y_train = train['是否违约']
            else:
                X_train = train
                y_train = None
            
            X_test = test.copy()
            
            # 创建特征工程系统
            fe_system = ComprehensiveFeatureEngineering(random_state=42)
            
            # 在训练集上拟合并转换
            print("在训练集上应用特征工程...")
            X_train_fe = fe_system.fit_transform(X_train, y_train)
            
            # 在测试集上转换
            print("在测试集上应用特征工程...")
            X_test_fe = fe_system.transform(X_test)
            
            # 保存转换器
            fe_system.save_transformers('comprehensive_fe_transformers.pkl')
            
            print(f"训练集特征工程: {X_train.shape[1]} -> {X_train_fe.shape[1]} 个特征")
            print(f"测试集特征工程: {X_test.shape[1]} -> {X_test_fe.shape[1]} 个特征")
            
            # 重建完整的数据集
            if y_train is not None:
                train_fe = pd.concat([X_train_fe, y_train], axis=1)
            else:
                train_fe = X_train_fe
            
            test_fe = X_test_fe
            
            result = {
                'train_fe': train_fe,
                'test_fe': test_fe,
                'fe_system': fe_system
            }
            
        except Exception as e:
            print(f"全面特征工程系统出错: {e}")
            print("回退到基础特征工程...")
            
            # 基础特征工程
            result = basic_feature_engineering(train, test)
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return engineer_features

def basic_feature_engineering(train, test):
    """基础特征工程"""
    print("使用基础特征工程...")
    
    # 1. 标签编码
    print("1. 标签编码...")
    label_encoders = {}
    categorical_cols = train.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col not in ['贷款ID']:  # 跳过ID列
            le = LabelEncoder()
            # 合并训练集和测试集的标签进行编码，避免测试集出现未知标签
            combined_values = pd.concat([train[col], test[col]], axis=0).astype(str)
            le.fit(combined_values)
            
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
            
            label_encoders[col] = le
    
    # 2. 特征组合
    print("2. 特征组合...")
    # 创建一些派生特征
    if '贷款总额' in train.columns and '月供' in train.columns:
        train['贷款收入比'] = train['贷款总额'] / (train['月供'] * 12 + 1)  # 避免除零
        test['贷款收入比'] = test['贷款总额'] / (test['月供'] * 12 + 1)
    
    if '信用卡数量' in train.columns and '信用卡总余额' in train.columns:
        train['平均信用卡余额'] = train['信用卡总余额'] / (train['信用卡数量'] + 1)
        test['平均信用卡余额'] = test['信用卡总余额'] / (test['信用卡数量'] + 1)
    
    # 3. 特征标准化
    print("3. 特征标准化...")
    scaler = StandardScaler()
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != '是否违约']  # 排除标签
    
    # 只在训练集上拟合scaler
    scaler.fit(train[numeric_cols])
    
    # 转换训练集和测试集
    train[numeric_cols] = scaler.transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])
    
    print(f"基础特征工程完成 - 训练集: {train.shape}, 测试集: {test.shape}")
    
    return {
        'train_fe': train,
        'test_fe': test,
        'scaler': scaler,
        'label_encoders': label_encoders
    }

def logistic_regression_stage():
    """Logistic Regression模型阶段 - 闭包函数"""
    def train_logistic_regression():
        print("=== Logistic Regression模型阶段 ===")
        
        # 检查依赖项
        dependencies = {
            'train_fe': global_data.get('train_fe'),
            'test_fe': global_data.get('test_fe')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'logistic_regression_model'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取特征工程后的数据
        train_fe = global_data['train_fe'].copy()
        test_fe = global_data['test_fe'].copy()
        
        print(f"Logistic Regression - 训练集: {train_fe.shape}, 测试集: {test_fe.shape}")
        
        # 准备数据
        X = train_fe.drop(columns=['是否违约', '贷款ID'])
        y = train_fe['是否违约']
        X_test = test_fe.drop(columns=['贷款ID'])
        
        print(f"特征维度: {X.shape}, 标签维度: {y.shape}")
        
        # Logistic Regression with GridSearchCV
        print("Logistic Regression with Cross-Validation...")
        lr_param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'class_weight': [None, 'balanced']
        }
        
        lr_grid = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            lr_param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        lr_grid.fit(X, y)
        
        print(f"Logistic Regression - Best AUC: {lr_grid.best_score_:.4f}")
        print(f"Best params: {lr_grid.best_params_}")
        
        # 特征重要性分析
        print("特征重要性分析...")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(lr_grid.best_estimator_.coef_[0])
        }).sort_values('importance', ascending=False)
        
        print(f"重要特征 (前10个):")
        print(feature_importance.head(10))
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('特征重要性')
        plt.title('Logistic Regression 特征重要性')
        plt.tight_layout()
        importance_path = '/Users/qingguo/Documents/project/carPricePredict/feature_importance_LogisticRegression.png'
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        result = {
            'lr_model': lr_grid.best_estimator_,
            'lr_cv_score': lr_grid.best_score_,
            'lr_best_params': lr_grid.best_params_,
            'lr_feature_importance': feature_importance
        }
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return train_logistic_regression

def random_forest_stage():
    """Random Forest模型阶段 - 闭包函数"""
    def train_random_forest():
        print("=== Random Forest模型阶段 ===")
        
        # 检查依赖项
        dependencies = {
            'train_fe': global_data.get('train_fe'),
            'test_fe': global_data.get('test_fe')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'random_forest_model'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取特征工程后的数据
        train_fe = global_data['train_fe'].copy()
        test_fe = global_data['test_fe'].copy()
        
        print(f"Random Forest - 训练集: {train_fe.shape}, 测试集: {test_fe.shape}")
        
        # 准备数据
        X = train_fe.drop(columns=['是否违约', '贷款ID'])
        y = train_fe['是否违约']
        X_test = test_fe.drop(columns=['贷款ID'])
        
        print(f"特征维度: {X.shape}, 标签维度: {y.shape}")
        
        # Random Forest with GridSearchCV
        print("Random Forest with Cross-Validation...")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'class_weight': [None, 'balanced']
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        rf_grid.fit(X, y)
        
        print(f"Random Forest - Best AUC: {rf_grid.best_score_:.4f}")
        print(f"Best params: {rf_grid.best_params_}")
        
        # 特征重要性分析
        print("特征重要性分析...")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_grid.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"重要特征 (前10个):")
        print(feature_importance.head(10))
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('特征重要性')
        plt.title('Random Forest 特征重要性')
        plt.tight_layout()
        importance_path = '/Users/qingguo/Documents/project/carPricePredict/feature_importance_RandomForest.png'
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        result = {
            'rf_model': rf_grid.best_estimator_,
            'rf_cv_score': rf_grid.best_score_,
            'rf_best_params': rf_grid.best_params_,
            'rf_feature_importance': feature_importance
        }
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return train_random_forest

def stacking_stage():
    """Stacking集成模型阶段 - 闭包函数"""
    def train_stacking():
        print("=== Stacking集成模型阶段 ===")
        
        # 检查依赖项 - 需要基础模型
        dependencies = {
            'train_fe': global_data.get('train_fe'),
            'test_fe': global_data.get('test_fe'),
            'lr_model': global_data.get('lr_model'),
            'rf_model': global_data.get('rf_model')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'stacking_model'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取特征工程后的数据
        train_fe = global_data['train_fe'].copy()
        test_fe = global_data['test_fe'].copy()
        
        print(f"Stacking - 训练集: {train_fe.shape}, 测试集: {test_fe.shape}")
        
        # 准备数据
        X = train_fe.drop(columns=['是否违约', '贷款ID'])
        y = train_fe['是否违约']
        
        print(f"特征维度: {X.shape}, 标签维度: {y.shape}")
        
        # 模型集成 - 堆叠法 (Stacking)
        print("模型集成 - Stacking...")
        from sklearn.ensemble import StackingClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        
        # 定义基础模型
        base_models = [
            ('lr', global_data['lr_model']),
            ('rf', global_data['rf_model']),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
        ]
        
        # 定义元模型
        meta_model = LogisticRegression(random_state=42)
        
        # 创建堆叠分类器
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,  # 使用5折交叉验证
            stack_method='predict_proba'  # 使用概率预测
        )
        
        # 训练堆叠模型
        stacking_model.fit(X, y)
        
        # 评估堆叠模型
        stacking_cv_score = cross_val_score(stacking_model, X, y, cv=3, scoring='roc_auc').mean()
        
        print(f"Stacking - AUC: {stacking_cv_score:.4f}")
        
        result = {
            'stacking_model': stacking_model,
            'stacking_cv_score': stacking_cv_score
        }
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return train_stacking

def voting_stage():
    """Voting集成模型阶段 - 闭包函数"""
    def train_voting():
        print("=== Voting集成模型阶段 ===")
        
        # 检查依赖项 - 需要基础模型和堆叠模型
        dependencies = {
            'train_fe': global_data.get('train_fe'),
            'test_fe': global_data.get('test_fe'),
            'lr_model': global_data.get('lr_model'),
            'rf_model': global_data.get('rf_model'),
            'stacking_model': global_data.get('stacking_model')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'voting_model'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取特征工程后的数据
        train_fe = global_data['train_fe'].copy()
        test_fe = global_data['test_fe'].copy()
        
        print(f"Voting - 训练集: {train_fe.shape}, 测试集: {test_fe.shape}")
        
        # 准备数据
        X = train_fe.drop(columns=['是否违约', '贷款ID'])
        y = train_fe['是否违约']
        
        print(f"特征维度: {X.shape}, 标签维度: {y.shape}")
        
        # 模型投票集成
        print("模型投票集成...")
        from sklearn.ensemble import VotingClassifier
        
        # 创建投票分类器
        voting_model = VotingClassifier(
            estimators=[
                ('lr', global_data['lr_model']),
                ('rf', global_data['rf_model']),
                ('stacking', global_data['stacking_model'])
            ],
            voting='soft'  # 使用软投票（基于概率）
        )
        
        # 训练投票模型
        voting_model.fit(X, y)
        
        # 评估投票模型
        voting_cv_score = cross_val_score(voting_model, X, y, cv=3, scoring='roc_auc').mean()
        
        print(f"Voting - AUC: {voting_cv_score:.4f}")
        
        result = {
            'voting_model': voting_model,
            'voting_cv_score': voting_cv_score
        }
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return train_voting

def model_comparison_stage():
    """模型比较和选择阶段 - 闭包函数"""
    def compare_models():
        print("=== 模型比较和选择阶段 ===")
        
        # 检查依赖项 - 需要所有模型
        dependencies = {
            'lr_cv_score': global_data.get('lr_cv_score'),
            'rf_cv_score': global_data.get('rf_cv_score'),
            'stacking_cv_score': global_data.get('stacking_cv_score'),
            'voting_cv_score': global_data.get('voting_cv_score')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'model_comparison'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 收集所有模型的性能
        cv_scores = {
            'LogisticRegression': global_data['lr_cv_score'],
            'RandomForest': global_data['rf_cv_score'],
            'Stacking': global_data['stacking_cv_score'],
            'Voting': global_data['voting_cv_score']
        }
        
        # 选择最佳模型
        best_model_name = max(cv_scores, key=cv_scores.get)
        
        print(f"最佳模型: {best_model_name} (AUC: {cv_scores[best_model_name]:.4f})")
        print("\n所有模型性能:")
        for model_name, score in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name}: {score:.4f}")
        
        # 模型性能对比
        performance_df = pd.DataFrame({
            'Model': list(cv_scores.keys()),
            'CV_AUC': list(cv_scores.values())
        }).sort_values('CV_AUC', ascending=False)
        
        print("\n模型交叉验证性能排名:")
        print(performance_df)
        
        # 绘制模型性能对比图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(performance_df['Model'], performance_df['CV_AUC'])
        plt.xlabel('模型')
        plt.ylabel('交叉验证AUC')
        plt.title('模型性能对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 绘制模型性能对比图（水平条形图）
        plt.subplot(1, 2, 2)
        plt.barh(performance_df['Model'], performance_df['CV_AUC'])
        plt.xlabel('交叉验证AUC')
        plt.ylabel('模型')
        plt.title('模型性能排名')
        
        plt.tight_layout()
        plt.savefig('/Users/qingguo/Documents/project/carPricePredict/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        result = {
            'best_model_name': best_model_name,
            'cv_scores': cv_scores,
            'performance_df': performance_df
        }
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return compare_models

def final_prediction_stage():
    """最终预测阶段 - 闭包函数"""
    def generate_final_predictions():
        print("=== 最终预测阶段 ===")
        
        # 检查依赖项
        dependencies = {
            'best_model_name': global_data.get('best_model_name'),
            'train_fe': global_data.get('train_fe'),
            'test_fe': global_data.get('test_fe'),
            'lr_model': global_data.get('lr_model'),
            'rf_model': global_data.get('rf_model'),
            'stacking_model': global_data.get('stacking_model'),
            'voting_model': global_data.get('voting_model')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'final_predictions'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取特征工程后的数据
        train_fe = global_data['train_fe'].copy()
        test_fe = global_data['test_fe'].copy()
        
        # 准备数据
        X = train_fe.drop(columns=['是否违约', '贷款ID'])
        y = train_fe['是否违约']
        X_test = test_fe.drop(columns=['贷款ID'])
        
        print(f"特征维度: {X.shape}, 标签维度: {y.shape}")
        
        # 根据最佳模型选择进行预测
        best_model_name = global_data['best_model_name']
        
        if best_model_name == 'LogisticRegression':
            best_model = global_data['lr_model']
        elif best_model_name == 'RandomForest':
            best_model = global_data['rf_model']
        elif best_model_name == 'Stacking':
            best_model = global_data['stacking_model']
        else:  # Voting
            best_model = global_data['voting_model']
        
        print(f"使用最佳模型: {best_model_name}")
        
        # 生成预测
        test_predictions = best_model.predict(X_test)
        test_probabilities = best_model.predict_proba(X_test)[:, 1]
        
        print(f"预测完成，预测数量: {len(test_predictions)}")
        print(f"预测概率分布:")
        print(f"预测概率均值: {test_probabilities.mean():.4f}")
        print(f"预测概率标准差: {test_probabilities.std():.4f}")
        print(f"预测概率最小值: {test_probabilities.min():.4f}")
        print(f"预测概率最大值: {test_probabilities.max():.4f}")
        
        # 绘制预测概率分布图
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(test_probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('预测概率')
        plt.ylabel('频次')
        plt.title('预测概率分布')
        plt.axvline(x=0.5, color='red', linestyle='--', label='决策阈值')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        prediction_counts = pd.Series(test_predictions).value_counts()
        plt.pie(prediction_counts.values, labels=['正常', '违约'], autopct='%1.1f%%', startangle=90)
        plt.title('预测结果分布')
        
        plt.tight_layout()
        plt.savefig('/Users/qingguo/Documents/project/carPricePredict/prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        result = {
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'best_model': best_model
        }
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return generate_final_predictions

def submission_stage():
    """生成提交文件阶段 - 闭包函数"""
    def generate_submission():
        print("\n=== 生成提交文件阶段 ===")
        
        # 检查依赖项
        dependencies = {
            'test_predictions': global_data.get('test_predictions'),
            'test_clean': global_data.get('test_clean')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        test_predictions = global_data['test_predictions']
        test_clean = global_data['test_clean']
        
        # 使用原始测试集的贷款ID字段
        loan_ids = test_clean['贷款ID']
        
        # 创建提交文件
        submission = pd.DataFrame({
            'ID': loan_ids,
            'label': test_predictions
        })
        
        # 保存提交文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_filename = f'25451354008submission_{timestamp}.csv'
        submission.to_csv(submission_filename, index=False)
        
        print(f"✓ 提交文件已生成: {submission_filename}")
        print(f"提交文件形状: {submission.shape}")
        print(f"预测结果分布:")
        print(submission['是否违约'].value_counts())
        
        return submission
    
    return generate_submission

def main():
    """主函数 - 执行完整的机器学习流程"""
    print("=== 贷款违约预测项目 ===")
    print(f"学生ID: {student_id}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 数据加载阶段
        print("\n" + "="*50)
        data_load_result = data_loading_stage()()
        if data_load_result is None:
            print("数据加载失败，程序终止")
            return
        
        # 更新全局数据
        global_data.update(data_load_result)
        
        # 2. 数据清洗阶段
        print("\n" + "="*50)
        data_clean_result = data_cleaning_stage()()
        if data_clean_result is None:
            print("数据清洗失败，程序终止")
            return
        
        # 更新全局数据
        global_data.update(data_clean_result)
        
        # 3. 探索性数据分析阶段
        print("\n" + "="*50)
        eda_result = eda_stage()()
        if eda_result is None:
            print("EDA失败，程序终止")
            return
        
        # 更新全局数据
        global_data.update(eda_result)
        
        # 4. 特征工程阶段
        print("\n" + "="*50)
        feature_engineering_result = feature_engineering_stage()()
        if feature_engineering_result is None:
            print("特征工程失败，程序终止")
            return
        
        # 更新全局数据
        global_data.update(feature_engineering_result)
        
        # 5. 建模阶段 - 使用独立的模型闭包函数
        print("\n" + "="*50)
        
        # 5.1 Logistic Regression模型
        print("\n" + "-"*30)
        lr_result = logistic_regression_stage()()
        if lr_result is None:
            print("Logistic Regression模型训练失败")
            return
        global_data.update(lr_result)
        
        # 5.2 Random Forest模型
        print("\n" + "-"*30)
        rf_result = random_forest_stage()()
        if rf_result is None:
            print("Random Forest模型训练失败")
            return
        global_data.update(rf_result)
        
        # 5.3 Stacking集成模型
        print("\n" + "-"*30)
        stacking_result = stacking_stage()()
        if stacking_result is None:
            print("Stacking集成模型训练失败")
            return
        global_data.update(stacking_result)
        
        # 5.4 Voting集成模型
        print("\n" + "-"*30)
        voting_result = voting_stage()()
        if voting_result is None:
            print("Voting集成模型训练失败")
            return
        global_data.update(voting_result)
        
        # 5.5 模型比较和选择
        print("\n" + "-"*30)
        comparison_result = model_comparison_stage()()
        if comparison_result is None:
            print("模型比较和选择失败")
            return
        global_data.update(comparison_result)
        
        # 5.6 最终预测
        print("\n" + "-"*30)
        prediction_result = final_prediction_stage()()
        if prediction_result is None:
            print("最终预测失败")
            return
        global_data.update(prediction_result)
        
        # 6. 提交文件生成阶段
        print("\n" + "="*50)
        submission_result = submission_stage()()
        if submission_result is None:
            print("提交文件生成失败")
            return
        
        # 更新全局数据
        global_data.update(submission_result)
        
        print("\n" + "="*50)
        print("=== 项目执行完成 ===")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总用时: {datetime.now() - start_time}")
        
        # 打印最终结果总结
        print(f"\n最终结果总结:")
        print(f"  最佳模型: {global_data['best_model_name']}")
        print(f"  最佳AUC: {global_data['cv_scores'][global_data['best_model_name']]:.4f}")
        print(f"  训练样本数: {len(global_data['train_fe'])}")
        print(f"  特征数: {global_data['train_fe'].shape[1]}")
        print(f"  测试预测数: {len(global_data['test_predictions'])}")
        
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return

# 如果作为主程序运行，执行main函数
if __name__ == "__main__":
    main()