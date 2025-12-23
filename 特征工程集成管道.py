#!/usr/bin/env python
# coding: utf-8

"""
特征工程集成管道
将全面的特征工程系统与现有的机器学习管道集成
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 导入全面特征工程系统
from 全面特征工程系统 import ComprehensiveFeatureEngineering

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class FeatureEngineeringPipeline:
    """特征工程集成管道"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_engineer = ComprehensiveFeatureEngineering(random_state=random_state)
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.cv_scores = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, train_path: str, test_path: str, target_col: str = 'isDefault'):
        """
        加载和准备数据
        """
        print("=== 数据加载和准备 ===")
        
        # 加载数据
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        print(f"训练数据形状: {train_data.shape}")
        print(f"测试数据形状: {test_data.shape}")
        
        # 分离特征和目标
        if target_col in train_data.columns:
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col]
        else:
            raise ValueError(f"目标列 '{target_col}' 不在训练数据中")
        
        # 保存测试数据ID（如果有）
        test_ids = None
        if 'id' in test_data.columns:
            test_ids = test_data['id']
            X_test = test_data.drop(columns=['id'])
        else:
            X_test = test_data
        
        return X_train, y_train, X_test, test_ids
    
    def apply_feature_engineering(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, save_path: str = None) -> tuple:
        """
        应用全面特征工程
        """
        print("\n=== 应用全面特征工程 ===")
        
        start_time = datetime.now()
        
        # 对训练数据应用特征工程
        print("处理训练数据...")
        X_train_engineered = self.feature_engineer.fit_transform(X_train, y_train)
        
        # 对测试数据应用相同的变换
        print("处理测试数据...")
        # 这里需要实现transform方法，或者重新拟合测试数据
        X_test_engineered = self.feature_engineer.fit_transform(X_test)
        
        # 标准化数值特征
        print("标准化数值特征...")
        numeric_cols = X_train_engineered.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            X_train_engineered[numeric_cols] = self.scaler.fit_transform(X_train_engineered[numeric_cols])
            X_test_engineered[numeric_cols] = self.scaler.transform(X_test_engineered[numeric_cols])
        
        # 处理可能的无穷值和NaN
        X_train_engineered = X_train_engineered.replace([np.inf, -np.inf], np.nan)
        X_test_engineered = X_test_engineered.replace([np.inf, -np.inf], np.nan)
        
        # 填充剩余的NaN值
        X_train_engineered = X_train_engineered.fillna(0)
        X_test_engineered = X_test_engineered.fillna(0)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"特征工程完成，耗时: {processing_time:.2f} 秒")
        print(f"训练数据: {X_train.shape} -> {X_train_engineered.shape}")
        print(f"测试数据: {X_test.shape} -> {X_test_engineered.shape}")
        
        # 保存特征工程结果
        if save_path:
            joblib.dump({
                'X_train': X_train_engineered,
                'X_test': X_test_engineered,
                'feature_engineer': self.feature_engineer,
                'scaler': self.scaler
            }, save_path)
            print(f"特征工程结果已保存到: {save_path}")
        
        return X_train_engineered, X_test_engineered
    
    def train_multiple_models_with_cv(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    cv_folds: int = 5) -> dict:
        """
        训练多个模型并进行交叉验证
        """
        print("\n=== 训练多个模型并进行交叉验证 ===")
        
        # 定义交叉验证策略
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # 定义要训练的模型
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        }
        
        # 定义超参数网格
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'class_weight': [None, 'balanced']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced']
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'scale_pos_weight': [1, 2, 5]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'scale_pos_weight': [1, 2, 5]
            }
        }
        
        best_models = {}
        cv_results = {}
        
        for model_name, model in models.items():
            print(f"\n训练 {model_name}...")
            
            # 超参数调优
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=cv_strategy, 
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 保存最佳模型
            best_models[model_name] = grid_search.best_estimator_
            
            # 交叉验证评分
            cv_scores = cross_val_score(
                grid_search.best_estimator_, 
                X_train, y_train, 
                cv=cv_strategy, 
                scoring='roc_auc'
            )
            
            cv_results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{model_name} 最佳参数: {grid_search.best_params_}")
            print(f"{model_name} 最佳CV分数: {grid_search.best_score_:.4f}")
            print(f"{model_name} CV分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = best_models
        self.cv_scores = cv_results
        
        return best_models, cv_results
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> VotingClassifier:
        """
        创建集成模型
        """
        print("\n=== 创建集成模型 ===")
        
        # 选择表现最好的3个模型进行集成
        model_performance = [(name, results['cv_mean']) for name, results in self.cv_scores.items()]
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        top_models = model_performance[:3]
        print(f"选择的最佳模型: {[name for name, _ in top_models]}")
        
        # 创建投票分类器
        ensemble_models = [(name, self.models[name]) for name, _ in top_models]
        
        voting_classifier = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # 使用概率投票
        )
        
        # 训练集成模型
        voting_classifier.fit(X_train, y_train)
        
        # 交叉验证评估集成模型
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        ensemble_cv_scores = cross_val_score(
            voting_classifier, X_train, y_train, 
            cv=cv_strategy, scoring='roc_auc'
        )
        
        print(f"集成模型CV分数: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
        
        self.models['Ensemble'] = voting_classifier
        self.cv_scores['Ensemble'] = {
            'cv_scores': ensemble_cv_scores,
            'cv_mean': ensemble_cv_scores.mean(),
            'cv_std': ensemble_cv_scores.std()
        }
        
        return voting_classifier
    
    def analyze_feature_importance(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        分析特征重要性
        """
        print("\n=== 特征重要性分析 ===")
        
        # 为每个模型分析特征重要性
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # 树模型特征重要性
                importance = model.feature_importances_
                feature_names = X_train.columns
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[model_name] = importance_df
                
                # 可视化前20个重要特征
                plt.figure(figsize=(10, 8))
                top_features = importance_df.head(20)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('特征重要性')
                plt.title(f'{model_name} - 特征重要性')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"{model_name} 前10个重要特征:")
                print(importance_df.head(10))
                
            elif hasattr(model, 'coef_'):
                # 线性模型系数
                coef = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                feature_names = X_train.columns
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coef
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[model_name] = importance_df
                
                print(f"{model_name} 前10个重要特征:")
                print(importance_df.head(10))
    
    def generate_predictions(self, X_test: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        生成预测
        """
        if model_name is None:
            # 使用最佳模型
            best_model_name = max(self.cv_scores.items(), key=lambda x: x[1]['cv_mean'])[0]
            model = self.models[best_model_name]
            print(f"使用最佳模型: {best_model_name}")
        else:
            model = self.models[model_name]
            print(f"使用模型: {model_name}")
        
        # 生成预测概率
        predictions = model.predict_proba(X_test)[:, 1]
        
        return predictions
    
    def save_submission(self, predictions: np.ndarray, test_ids: pd.Series, 
                         submission_path: str, model_name: str = None):
        """
        保存提交文件
        """
        submission = pd.DataFrame({
            'id': test_ids,
            'isDefault': predictions
        })
        
        submission.to_csv(submission_path, index=False)
        
        model_used = model_name if model_name else max(self.cv_scores.items(), key=lambda x: x[1]['cv_mean'])[0]
        print(f"使用 {model_used} 模型的预测结果已保存到: {submission_path}")
        
        # 显示预测分布
        plt.figure(figsize=(10, 6))
        plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('违约概率')
        plt.ylabel('频次')
        plt.title(f'{model_used} - 预测分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{model_used.lower().replace(" ", "_")}_prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_pipeline(self, train_path: str, test_path: str, 
                            target_col: str = 'isDefault', 
                            output_dir: str = './output'):
        """
        运行完整的机器学习管道
        """
        print("=== 运行完整机器学习管道 ===")
        start_time = datetime.now()
        
        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 加载数据
        X_train, y_train, X_test, test_ids = self.load_and_prepare_data(
            train_path, test_path, target_col
        )
        
        # 2. 特征工程
        X_train_fe, X_test_fe = self.apply_feature_engineering(
            X_train, y_train, X_test,
            save_path=f'{output_dir}/feature_engineering_results.pkl'
        )
        
        # 3. 训练多个模型
        best_models, cv_results = self.train_multiple_models_with_cv(X_train_fe, y_train)
        
        # 4. 创建集成模型
        ensemble_model = self.create_ensemble_model(X_train_fe, y_train)
        
        # 5. 特征重要性分析
        self.analyze_feature_importance(X_train_fe, y_train)
        
        # 6. 生成预测
        predictions = self.generate_predictions(X_test_fe)
        
        # 7. 保存提交文件
        self.save_submission(predictions, test_ids, f'{output_dir}/submission.csv')
        
        # 8. 保存模型和结果
        joblib.dump({
            'models': self.models,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance,
            'feature_engineer': self.feature_engineer
        }, f'{output_dir}/complete_pipeline_results.pkl')
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\n=== 管道运行完成 ===")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"输出目录: {output_dir}")
        
        # 显示最终结果总结
        self.print_final_summary()
        
        return predictions
    
    def print_final_summary(self):
        """
        打印最终结果总结
        """
        print("\n=== 最终结果总结 ===")
        
        # 模型性能总结
        print("模型交叉验证性能:")
        for model_name, results in self.cv_scores.items():
            print(f"  {model_name}: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        
        # 最佳模型
        best_model_name = max(self.cv_scores.items(), key=lambda x: x[1]['cv_mean'])[0]
        print(f"\n最佳模型: {best_model_name}")
        print(f"最佳CV分数: {self.cv_scores[best_model_name]['cv_mean']:.4f}")
        
        # 特征工程总结
        if hasattr(self.feature_engineer, 'feature_info'):
            print(f"\n特征工程效果:")
            print(f"  数值特征: {len(self.feature_engineer.numerical_features)}")
            print(f"  分类特征: {len(self.feature_engineer.categorical_features)}")
            print(f"  时间特征: {len(self.feature_engineer.datetime_features)}")


def main():
    """主函数"""
    print("开始运行特征工程集成管道...")
    
    # 创建管道实例
    pipeline = FeatureEngineeringPipeline(random_state=42)
    
    # 运行完整管道
    predictions = pipeline.run_complete_pipeline(
        train_path='/Users/qingguo/Documents/project/carPricePredict/data/train.csv',
        test_path='/Users/qingguo/Documents/project/carPricePredict/data/test.csv',
        target_col='isDefault',
        output_dir='/Users/qingguo/Documents/project/carPricePredict/output_comprehensive'
    )
    
    print("特征工程集成管道运行完成！")


if __name__ == "__main__":
    main()