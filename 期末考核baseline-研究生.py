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
import pickle
import hashlib
import os
from datetime import datetime
from typing import Dict, Any, Tuple
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

# 缓存目录设置
CACHE_DIR = '/Users/qingguo/Documents/project/carPricePredict/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(stage_name: str) -> str:
    """获取缓存文件路径"""
    return os.path.join(CACHE_DIR, f'stage_{stage_name}.pkl')

def get_dependency_hash(dependencies: Dict[str, Any]) -> str:
    """计算依赖项的哈希值"""
    # 将依赖项转换为字符串并计算哈希
    dep_str = str(sorted(dependencies.items()))
    return hashlib.md5(dep_str.encode()).hexdigest()

def load_cache(stage_name: str, dependencies: Dict[str, Any]) -> Any:
    """加载缓存数据"""
    cache_path = get_cache_path(stage_name)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # 检查依赖项是否发生变化
        cached_hash = cached_data.get('dependency_hash', '')
        current_hash = get_dependency_hash(dependencies)
        
        if cached_hash == current_hash:
            print(f"✓ {stage_name}: 使用缓存数据")
            return cached_data['data']
        else:
            print(f"✗ {stage_name}: 依赖项发生变化，重新计算")
            return None
    except Exception as e:
        print(f"✗ {stage_name}: 缓存加载失败 - {e}")
        return None

def save_cache(stage_name: str, data: Any, dependencies: Dict[str, Any]) -> None:
    """保存缓存数据"""
    cache_path = get_cache_path(stage_name)
    
    try:
        cache_data = {
            'data': data,
            'dependency_hash': get_dependency_hash(dependencies),
            'timestamp': datetime.now()
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✓ {stage_name}: 缓存已保存")
    except Exception as e:
        print(f"✗ {stage_name}: 缓存保存失败 - {e}")

def check_dependencies(dependencies: Dict[str, Any]) -> bool:
    """检查依赖项是否存在"""
    for dep_name, dep_value in dependencies.items():
        # 使用is None检查，避免DataFrame的布尔评估问题
        if dep_value is None:
            print(f"✗ 依赖项缺失: {dep_name}")
            return False
    return True

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

def modeling_stage():
    """建模阶段 - 闭包函数"""
    def build_models():
        print("=== 建模阶段 ===")
        
        # 检查依赖项
        dependencies = {
            'train_fe': global_data.get('train_fe'),
            'test_fe': global_data.get('test_fe')
        }
        
        if not check_dependencies(dependencies):
            return None
        
        # 检查缓存
        cache_key = 'modeling'
        cached_result = load_cache(cache_key, dependencies)
        if cached_result is not None:
            return cached_result
        
        # 获取特征工程后的数据
        train_fe = global_data['train_fe'].copy()
        test_fe = global_data['test_fe'].copy()
        
        print(f"建模 - 训练集: {train_fe.shape}, 测试集: {test_fe.shape}")
        
        # 1. 准备数据
        print("1. 数据准备...")
        # 分离特征和目标
        X = train_fe.drop(columns=['是否违约', '贷款ID'])
        y = train_fe['是否违约']
        
        # 测试集特征（无标签）
        X_test = test_fe.drop(columns=['贷款ID'])
        
        print(f"特征维度: {X.shape}, 标签维度: {y.shape}")
        
        # 2. 模型训练与交叉验证
        print("2. 模型训练与交叉验证...")
        
        # 定义模型和超参数空间
        models = {}
        cv_scores = {}
        
        # Logistic Regression with GridSearchCV
        print("2.1 Logistic Regression with Cross-Validation...")
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
        models['LogisticRegression'] = lr_grid.best_estimator_
        cv_scores['LogisticRegression'] = lr_grid.best_score_
        
        print(f"Logistic Regression - Best AUC: {lr_grid.best_score_:.4f}")
        print(f"Best params: {lr_grid.best_params_}")
        
        # Random Forest with GridSearchCV
        print("2.2 Random Forest with Cross-Validation...")
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
        models['RandomForest'] = rf_grid.best_estimator_
        cv_scores['RandomForest'] = rf_grid.best_score_
        
        print(f"Random Forest - Best AUC: {rf_grid.best_score_:.4f}")
        print(f"Best params: {rf_grid.best_params_}")
        
        # 3. 模型集成 - 堆叠法 (Stacking)
        print("2.3 模型集成 - Stacking...")
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import GradientBoostingClassifier
        
        # 定义基础模型
        base_models = [
            ('lr', LogisticRegression(C=1, penalty='l2', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
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
        
        models['Stacking'] = stacking_model
        cv_scores['Stacking'] = stacking_cv_score
        
        print(f"Stacking - AUC: {stacking_cv_score:.4f}")
        
        # 模型投票集成
        print("2.4 模型投票集成...")
        from sklearn.ensemble import VotingClassifier
        
        # 创建投票分类器
        voting_model = VotingClassifier(
            estimators=[
                ('lr', lr_grid.best_estimator_),
                ('rf', rf_grid.best_estimator_),
                ('stacking', stacking_model)
            ],
            voting='soft'  # 使用软投票（基于概率）
        )
        
        # 训练投票模型
        voting_model.fit(X, y)
        
        # 评估投票模型
        voting_cv_score = cross_val_score(voting_model, X, y, cv=3, scoring='roc_auc').mean()
        
        models['Voting'] = voting_model
        cv_scores['Voting'] = voting_cv_score
        
        print(f"Voting - AUC: {voting_cv_score:.4f}")
        
        # 3. 选择最佳模型
        print("3. 选择最佳模型...")
        best_model_name = max(cv_scores, key=cv_scores.get)
        best_model = models[best_model_name]
        
        print(f"最佳模型: {best_model_name} (AUC: {cv_scores[best_model_name]:.4f})")
        print("\n所有模型性能:")
        for model_name, score in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name}: {score:.4f}")
        
        # 4. 模型性能对比
        print("\n4. 模型性能对比...")
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
        
        # 5. 特征重要性分析（仅对单一模型）
        print("\n5. 特征重要性分析...")
        
        # 获取最佳模型的特征重要性
        if hasattr(best_model, 'feature_importances_'):
            # 树模型特征重要性
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n{best_model_name} 重要特征 (前10个):")
            print(feature_importance.head(10))
            
            # 绘制特征重要性图
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(15)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('特征重要性')
            plt.title(f'{best_model_name} 特征重要性')
            plt.tight_layout()
            importance_path = f'/Users/qingguo/Documents/project/carPricePredict/feature_importance_{best_model_name}.png'
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        elif hasattr(best_model, 'coef_'):
            # 线性模型特征重要性
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(best_model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            print(f"\n{best_model_name} 重要特征 (前10个):")
            print(feature_importance.head(10))
            
            # 绘制特征重要性图
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(15)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('特征重要性')
            plt.title(f'{best_model_name} 特征重要性')
            plt.tight_layout()
            importance_path = f'/Users/qingguo/Documents/project/carPricePredict/feature_importance_{best_model_name}.png'
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        return best_model
        
        # 6. 预测生成
        print("6. 预测生成...")
        
        # 使用最佳模型进行预测
        test_predictions = best_model.predict_proba(X_test)[:, 1]
        
        # 创建提交文件
        submission = pd.DataFrame({
            '贷款ID': test_df['贷款ID'],
            '违约概率': test_predictions
        })
        
        # 保存提交文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_filename = f"{student_id}submission_{timestamp}.csv"
        submission.to_csv(submission_filename, index=False)
        
        print(f"\n提交文件已保存: {submission_filename}")
        print(f"预测结果分布:")
        print(f"  违约概率 < 0.5: {(test_predictions < 0.5).sum()}")
        print(f"  违约概率 >= 0.5: {(test_predictions >= 0.5).sum()}")
        
        # 7. 结果总结
        print("\n7. 结果总结:")
        print(f"  训练样本数: {len(X)}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  模型数: {len(models)}")
        print(f"  最佳模型: {best_model_name}")
        print(f"  最佳AUC: {cv_scores[best_model_name]:.4f}")
        print(f"  测试预测数: {len(test_predictions)}")
        
        return best_model, submission
        
        print(f"\n预测概率分布:")
        print(f"预测概率均值: {test_probabilities.mean():.4f}")
        print(f"预测概率标准差: {test_probabilities.std():.4f}")
        print(f"预测概率最小值: {test_probabilities.min():.4f}")
        print(f"预测概率最大值: {test_probabilities.max():.4f}")
        
        # 创建提交文件
        submission = pd.DataFrame({
            '贷款ID': test_df['贷款ID'],
            '违约概率': test_probabilities
        })
        
        # 保存提交文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_filename = f"{student_id}submission_{timestamp}.csv"
        submission.to_csv(submission_filename, index=False)
        
        print(f"\n提交文件已保存: {submission_filename}")
        print(f"提交文件形状: {submission.shape}")
        
        return best_model
        
        print(f"预测结果分布:")
        print(f"违约预测数量: {sum(test_predictions)}")
        print(f"违约预测比例: {sum(test_predictions) / len(test_predictions) * 100:.2f}%")
        
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
            'train_features': X,
            'test_features': X_test,
            'train_target': y,
            'models': models,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'cv_scores': cv_scores,
            'performance_df': performance_df
        }
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return build_models

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
    """主执行函数"""
    print("开始执行增强版汽车价格预测基线...")
    print(f"缓存目录: {CACHE_DIR}")
    
    # 阶段1: 数据加载
    print("\n=== 阶段0: 数据加载 ===")
    try:
        train = pd.read_csv('/Users/qingguo/Documents/project/carPricePredict/data/train.csv')
        test = pd.read_csv('/Users/qingguo/Documents/project/carPricePredict/data/testA.csv')
        
        global_data['train_raw'] = train
        global_data['test_raw'] = test
        
        print(f"数据加载成功 - 训练集: {train.shape}, 测试集: {test.shape}")
        print(f"训练集列名: {list(train.columns)}")
        print(f"测试集列名: {list(test.columns)}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 阶段1: 数据清洗
    cleaning_func = data_cleaning_stage()
    cleaning_result = cleaning_func()
    if cleaning_result is not None:
        global_data.update(cleaning_result)
    else:
        print("数据清洗阶段失败")
        return
    
    # 阶段2: 探索性数据分析
    eda_func = eda_stage()
    eda_result = eda_func()
    if eda_result is not None:
        global_data.update(eda_result)
    else:
        print("EDA阶段失败")
        return
    
    # 阶段3: 特征工程
    fe_func = feature_engineering_stage()
    fe_result = fe_func()
    if fe_result is not None:
        global_data.update(fe_result)
    else:
        print("特征工程阶段失败")
        return
    
    # 阶段4: 建模
    modeling_func = modeling_stage()
    modeling_result = modeling_func()
    if modeling_result is not None:
        global_data.update(modeling_result)
    else:
        print("建模阶段失败")
        return
    
    # 阶段5: 生成提交文件
    submission_func = submission_stage()
    submission_result = submission_func()
    if submission_result is not None:
        global_data['submission'] = submission_result
        print("\n=== 执行完成 ===")
        print(f"✓ 所有阶段执行成功")
        print(f"✓ 提交文件已生成")
        print(f"✓ 缓存已保存")
        print(f"✓ 可视化图表已生成")
        print("\n总结:")
        print(f"- 数据清洗: {global_data['train_clean'].shape[0]} 条训练样本")
        print(f"- 特征工程: {global_data['train_fe'].shape[1]} 个特征")
        print(f"- 模型数量: {len(global_data['models'])}")
        print(f"- 最佳模型: {global_data['best_model_name']}")
        print(f"- 测试集预测数量: {len(global_data['test_predictions'])}")
        print("="*60)
    else:
        print("提交文件生成阶段失败")

# 如果作为主程序运行，执行main函数
if __name__ == "__main__":
    main()