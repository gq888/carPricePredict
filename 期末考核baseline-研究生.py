#!/usr/bin/env python
# coding: utf-8

"""
增强版汽车价格预测基线脚本
实现了四个核心阶段的模块化封装，包含缓存机制和依赖检查
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
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
        print("\n=== 探索性数据分析阶段 ===")
        
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
        
        train = global_data['train_clean']
        test = global_data['test_clean']
        
        print("1. 基础统计分析...")
        # 数值特征描述性统计
        numeric_features = train.select_dtypes(include=[np.number]).columns
        print(f"数值特征数量: {len(numeric_features)}")
        print("数值特征统计:")
        print(train[numeric_features].describe())
        
        # 分类特征分析
        categorical_features = train.select_dtypes(include=['object']).columns
        print(f"\n分类特征数量: {len(categorical_features)}")
        for col in categorical_features[:5]:  # 只显示前5个分类特征
            print(f"\n{col} - 唯一值数量: {train[col].nunique()}")
            print(train[col].value_counts().head())
        
        print("\n2. 目标变量分析...")
        if '是否违约' in train.columns:
            target_dist = train['是否违约'].value_counts()
            print("目标变量分布:")
            print(target_dist)
            print(f"违约率: {target_dist[1] / len(train) * 100:.2f}%")
            
            # 创建目标变量分布图 - 智能处理中文显示
            plt.figure(figsize=(10, 6))
            
            # 检测是否使用中文字体
            current_font = plt.rcParams['font.sans-serif'][0]
            use_chinese = current_font not in ['DejaVu Sans', 'Arial', 'Helvetica']
            
            plt.subplot(1, 2, 1)
            train['是否违约'].value_counts().plot(kind='bar')
            if use_chinese:
                plt.title('目标变量分布')
                plt.xlabel('是否违约')
                plt.ylabel('数量')
            else:
                plt.title('Target Variable Distribution')
                plt.xlabel('Default Status')
                plt.ylabel('Count')
            
            plt.subplot(1, 2, 2)
            if use_chinese:
                plt.pie(train['是否违约'].value_counts().values, 
                       labels=['正常', '违约'], autopct='%1.1f%%')
                plt.title('目标变量比例')
            else:
                plt.pie(train['是否违约'].value_counts().values, 
                       labels=['Normal', 'Default'], autopct='%1.1f%%')
                plt.title('Target Variable Proportion')
            
            plt.tight_layout()
            plt.savefig('/Users/qingguo/Documents/project/carPricePredict/eda_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✓ EDA可视化已保存")
        
        result = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'target_distribution': target_dist if '是否违约' in train.columns else None
        }
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return perform_eda

def feature_engineering_stage():
    """特征工程阶段 - 闭包函数"""
    def engineer_features():
        print("\n=== 特征工程阶段 ===")
        
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
        
        train = global_data['train_clean'].copy()
        test = global_data['test_clean'].copy()
        
        print("1. 特征编码...")
        # 对分类特征进行标签编码
        categorical_cols = train.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            # 合并训练集和测试集的唯一值进行编码
            all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
            le.fit(all_values)
            
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
        
        print("2. 特征构造...")
        # 构造新特征
        # 2.1 贷款相关特征
        if '贷款总额' in train.columns and '月还款额' in train.columns:
            train['还款比例'] = train['月还款额'] / train['贷款总额']
            test['还款比例'] = test['月还款额'] / test['贷款总额']
        
        # 2.2 信用评分相关特征
        if '信用评分低值' in train.columns and '信用评分高值' in train.columns:
            train['信用评分范围'] = train['信用评分高值'] - train['信用评分低值']
            test['信用评分范围'] = test['信用评分高值'] - test['信用评分低值']
            
            train['信用评分均值'] = (train['信用评分低值'] + train['信用评分高值']) / 2
            test['信用评分均值'] = (test['信用评分低值'] + test['信用评分高值']) / 2
        
        # 2.3 负债收入比相关特征
        if '负债收入比' in train.columns:
            # 创建负债收入ospin桶
            train['负债收入比等级'] = pd.cut(train['负债收入比'], 
                                         bins=[0, 10, 20, 30, 100], 
                                         labels=[0, 1, 2, 3])
            test['负债收入比等级'] = pd.cut(test['负债收入比'], 
                                         bins=[0, 10, 20, 30, 100], 
                                         labels=[0, 1, 2, 3])
        
        # 2.4 工作年限相关特征
        if '工作年限' in train.columns:
            # 将工作年限转换为数值
            work_year_map = {
                '< 1 year': 0,
                '1 year': 1,
                '2 years': 2,
                '3 years': 3,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '7 years': 7,
                '8 years': 8,
                '9 years': 9,
                '10+ years': 10
            }
            
            train['工作年限数值'] = train['工作年限'].map(work_year_map).fillna(5)
            test['工作年限数值'] = test['工作年限'].map(work_year_map).fillna(5)
        
        print("3. 特征选择...")
        # 移除低方差特征
        from sklearn.feature_selection import VarianceThreshold
        
        # 分离特征和标签
        if '是否违约' in train.columns:
            X_train = train.drop(['是否违约'], axis=1)
            y_train = train['是否违约']
        else:
            X_train = train
            y_train = None
        
        X_test = test
        
        # 应用方差阈值
        selector = VarianceThreshold(threshold=0.01)
        X_train_selected = selector.fit_transform(X_train)
        X_test_selected = selector.transform(X_test)
        
        # 获取选择的特征名称
        selected_features = X_train.columns[selector.get_support()]
        
        print(f"原始特征数量: {X_train.shape[1]}")
        print(f"选择后特征数量: {len(selected_features)}")
        
        # 重新构建DataFrame
        train_fe = pd.DataFrame(X_train_selected, columns=selected_features, index=train.index)
        test_fe = pd.DataFrame(X_test_selected, columns=selected_features, index=test.index)
        
        if y_train is not None:
            train_fe['是否违约'] = y_train
        
        print("4. 最终缺失值处理...")
        # 确保没有缺失值
        train_fe = train_fe.fillna(0)
        test_fe = test_fe.fillna(0)
        
        # 确保所有数值都是有限的
        train_fe = train_fe.replace([np.inf, -np.inf], 0)
        test_fe = test_fe.replace([np.inf, -np.inf], 0)
        
        print(f"特征工程完成 - 训练集: {train_fe.shape}, 测试集: {test_fe.shape}")
        
        result = {'train_fe': train_fe, 'test_fe': test_fe}
        
        # 保存缓存
        save_cache(cache_key, result, dependencies)
        
        return result
    
    return engineer_features

def modeling_stage():
    """数据建模阶段 - 闭包函数"""
    def build_models():
        print("\n=== 数据建模阶段 ===")
        
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
        
        train_fe = global_data['train_fe']
        test_fe = global_data['test_fe']
        
        print("1. 数据标准化...")
        scaler = StandardScaler()
        
        # 分离特征和标签
        X_train = train_fe.drop(['是否违约'], axis=1)
        y_train = train_fe['是否违约']
        X_test = test_fe
        
        # 确保没有缺失值和无限值
        X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
        X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("2. 模型训练...")
        models = {}
        
        # 2.1 逻辑回归模型
        print("训练逻辑回归模型...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        models['LogisticRegression'] = lr_model
        
        # 2.2 交叉验证评估
        print("3. 模型评估...")
        cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"逻辑回归交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 2.3 特征重要性分析
        if hasattr(lr_model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(lr_model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            print("重要特征 (前10个):")
            print(feature_importance.head(10))
            
            # 绘制特征重要性图
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(15)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('特征重要性')
            plt.title('逻辑回归特征重要性')
            plt.tight_layout()
            plt.savefig('/Users/qingguo/Documents/project/carPricePredict/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("4. 预测生成...")
        # 在测试集上进行预测
        test_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]
        test_predictions = lr_model.predict(X_test_scaled)
        
        print(f"预测结果统计:")
        print(f"违约预测数量: {sum(test_predictions)}")
        print(f"违约预测比例: {sum(test_predictions) / len(test_predictions) * 100:.2f}%")
        
        result = {
            'train_scaled': pd.DataFrame(X_train_scaled, columns=X_train.columns, index=train_fe.index),
            'test_scaled': pd.DataFrame(X_test_scaled, columns=X_train.columns, index=test_fe.index),
            'models': models,
            'best_model': lr_model,
            'best_model_name': 'LogisticRegression',
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'scaler': scaler
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
            'test_probabilities': global_data.get('test_probabilities'),
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
            'id': loan_ids,
            '是否违约': test_predictions
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
    
    # 阶段4: 数据建模
    modeling_func = modeling_stage()
    modeling_result = modeling_func()
    if modeling_result is not None:
        global_data.update(modeling_result)
    else:
        print("数据建模阶段失败")
        return
    
    # 阶段5: 生成提交文件
    submission_func = submission_stage()
    submission_result = submission_func()
    if submission_result is not None:
        global_data['submission'] = submission_result
        print("✓ 所有阶段执行完成！")
    else:
        print("提交文件生成失败")
        return
    
    # 最终总结
    print("\n" + "="*60)
    print("增强版汽车价格预测基线执行完成！")
    print("="*60)
    print(f"最终数据形状:")
    print(f"- 清洗后训练集: {global_data['train_clean'].shape}")
    print(f"- 特征工程训练集: {global_data['train_fe'].shape}")
    print(f"- 标准化训练集: {global_data['train_scaled'].shape}")
    print(f"- 模型数量: {len(global_data['models'])}")
    print(f"- 最佳模型: {global_data['best_model_name']}")
    print(f"- 测试集预测数量: {len(global_data['test_predictions'])}")
    print("="*60)

# 如果作为主程序运行，执行main函数
if __name__ == "__main__":
    main()