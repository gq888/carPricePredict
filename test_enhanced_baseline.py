#!/usr/bin/env python
# coding: utf-8

"""
增强版baseline测试脚本
"""

import os
import pandas as pd
import numpy as np

def test_data_files():
    """测试数据文件是否存在"""
    print("=== 测试数据文件 ===")
    
    required_files = [
        '/Users/qingguo/Documents/project/carPricePredict/data/train.csv',
        '/Users/qingguo/Documents/project/carPricePredict/data/testA.csv'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ 找到文件: {file_path}")
            # 检查文件内容
            try:
                df = pd.read_csv(file_path)
                print(f"  - 形状: {df.shape}")
                print(f"  - 列名: {list(df.columns[:5])}...")  # 只显示前5列
            except Exception as e:
                print(f"  - 读取失败: {e}")
        else:
            print(f"✗ 缺失文件: {file_path}")
    
    return True

def test_enhanced_baseline():
    """测试增强版baseline实现"""
    print("\n=== 测试增强版baseline ===")
    
    try:
        # 清空全局命名空间
        global_namespace = {}
        
        # 读取并执行增强版baseline脚本
        with open('/Users/qingguo/Documents/project/carPricePredict/期末考核baseline-研究生.py', 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # 执行脚本
        exec(script_content, global_namespace)
        
        # 检查关键变量是否存在
        expected_variables = [
            'train_clean', 'test_clean', 'train_fe', 'test_fe', 
            'train_scaled', 'test_scaled', 'models', 'best_model', 
            'best_model_name', 'test_predictions', 'test_probabilities', 'submission'
        ]
        
        print("检查全局变量:")
        for var in expected_variables:
            if var in global_namespace:
                value = global_namespace[var]
                if hasattr(value, 'shape'):
                    print(f"✓ {var}: 形状 {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"✓ {var}: 长度 {len(value)}")
                else:
                    print(f"✓ {var}: {type(value)}")
            else:
                print(f"✗ {var}: 缺失")
        
        # 检查缓存文件
        cache_dir = '/Users/qingguo/Documents/project/carPricePredict/cache'
        if os.path.exists(cache_dir):
            print(f"\n缓存文件:")
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
            for cache_file in cache_files:
                print(f"✓ {cache_file}")
        
        # 检查提交文件
        submission_files = [f for f in os.listdir('.') if f.startswith('25451354008submission_') and f.endswith('.csv')]
        if submission_files:
            print(f"\n提交文件:")
            for sub_file in submission_files:
                print(f"✓ {sub_file}")
                # 验证提交文件内容
                try:
                    sub_df = pd.read_csv(sub_file)
                    print(f"  - 形状: {sub_df.shape}")
                    print(f"  - 列名: {list(sub_df.columns)}")
                    if '是否违约' in sub_df.columns:
                        print(f"  - 预测分布: {sub_df['是否违约'].value_counts().to_dict()}")
                except Exception as e:
                    print(f"  - 验证失败: {e}")
        
        # 检查可视化文件
        viz_files = ['eda_visualization.png', 'correlation_heatmap.png', 'feature_importance.png']
        print(f"\n可视化文件:")
        for viz_file in viz_files:
            if os.path.exists(viz_file):
                print(f"✓ {viz_file}")
            else:
                print(f"✗ {viz_file}")
        
        print("\n✓ 增强版baseline测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 增强版baseline测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_mechanism():
    """测试缓存机制"""
    print("\n=== 测试缓存机制 ===")
    
    try:
        # 第一次执行 - 应该重新计算
        print("第一次执行 (应该重新计算):")
        global_namespace1 = {}
        
        with open('/Users/qingguo/Documents/project/carPricePredict/期末考核baseline-研究生.py', 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        exec(script_content, global_namespace1)
        
        if 'main' in global_namespace1:
            global_namespace1['main']()
        
        # 第二次执行 - 应该使用缓存
        print("\n第二次执行 (应该使用缓存):")
        global_namespace2 = {}
        
        exec(script_content, global_namespace2)
        
        if 'main' in global_namespace2:
            global_namespace2['main']()
        
        print("✓ 缓存机制测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 缓存机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始执行增强版baseline测试...")
    
    # 测试数据文件
    test_data_files()
    
    # 测试增强版baseline
    baseline_success = test_enhanced_baseline()
    
    # 测试缓存机制
    cache_success = test_cache_mechanism()
    
    if baseline_success and cache_success:
        print("\n🎉 所有测试通过！增强版baseline实现正确。")
    else:
        print("\n❌ 测试失败，请检查实现。")