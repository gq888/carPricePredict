
import pickle
import hashlib
import os
import json
from typing import Dict, Any, Tuple
from datetime import datetime

# 缓存目录设置
CACHE_DIR = '/Users/qingguo/Documents/project/carPricePredict/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(stage_name: str) -> str:
    """获取缓存文件路径"""
    return os.path.join(CACHE_DIR, f'stage_{stage_name}.pkl')

def get_dependency_hash(dependencies: Dict[str, Any]) -> str:
    """计算依赖项的哈希值"""
    # 创建一个可序列化的依赖项表示
    serializable_deps = {}
    
    for key, value in sorted(dependencies.items()):
        if value is None:
            serializable_deps[key] = None
        elif hasattr(value, 'shape'):  # DataFrame、Series或numpy数组
            # 对于DataFrame，使用形状、列名和数据的稳定哈希作为标识
            if hasattr(value, 'columns') and len(value.columns) > 1:  # DataFrame
                # 使用稳定的特征进行哈希：形状、列名、数据类型摘要
                data_summary = []
                for col in sorted(value.columns):
                    col_data = value[col]
                    if len(col_data) > 0:
                        # 计算列数据的稳定摘要（前中后采样）
                        sample_size = min(10, len(col_data))
                        front = col_data.iloc[:sample_size].tolist()
                        middle = col_data.iloc[len(col_data)//2:len(col_data)//2+sample_size].tolist() if len(col_data) > sample_size else front
                        back = col_data.iloc[-sample_size:].tolist() if len(col_data) > sample_size else front
                        data_summary.append({
                            'col': col,
                            'dtype': str(col_data.dtype),
                            'samples': [front, middle, back]
                        })
                
                serializable_deps[key] = {
                    'type': 'DataFrame',
                    'shape': value.shape,
                    'columns': sorted(list(value.columns)),  # 排序确保一致性
                    'data_summary': data_summary
                }
            elif hasattr(value, 'name') or (hasattr(value, 'columns') and len(value.columns) == 1):  # Series
                # 对于Series，使用名称、形状和数据采样
                series_name = getattr(value, 'name', 'unnamed')
                if len(value) > 0:
                    sample_size = min(20, len(value))
                    front = value.iloc[:sample_size].tolist()
                    middle = value.iloc[len(value)//2:len(value)//2+sample_size].tolist() if len(value) > sample_size else front
                    back = value.iloc[-sample_size:].tolist() if len(value) > sample_size else front
                    
                    serializable_deps[key] = {
                        'type': 'Series',
                        'shape': value.shape,
                        'name': series_name,
                        'dtype': str(value.dtype),
                        'samples': [front, middle, back]
                    }
                else:
                    serializable_deps[key] = {
                        'type': 'Series',
                        'shape': value.shape,
                        'name': series_name,
                        'dtype': str(value.dtype),
                        'samples': []
                    }
            else:  # numpy数组
                # 对于numpy数组，使用形状、dtype和前中后采样
                if value.size > 0:
                    sample_size = min(100, value.size)
                    flat = value.flatten()
                    front = flat[:sample_size].tolist()
                    middle = flat[len(flat)//2:len(flat)//2+sample_size].tolist() if len(flat) > sample_size else front
                    back = flat[-sample_size:].tolist() if len(flat) > sample_size else front
                    
                    serializable_deps[key] = {
                        'type': 'ndarray',
                        'shape': value.shape,
                        'dtype': str(value.dtype),
                        'samples': [front, middle, back]
                    }
                else:
                    serializable_deps[key] = {
                        'type': 'ndarray',
                        'shape': value.shape,
                        'dtype': str(value.dtype),
                        'samples': []
                    }
        elif isinstance(value, (str, int, float, bool)):
            serializable_deps[key] = value
        elif isinstance(value, (list, tuple)):
            serializable_deps[key] = {'type': type(value).__name__, 'data': list(value)}
        elif isinstance(value, dict):
            serializable_deps[key] = {'type': 'dict', 'data': dict(sorted(value.items()))}
        else:
            # 对于其他复杂对象，使用其字符串表示和属性
            serializable_deps[key] = {
                'type': type(value).__name__,
                'str_repr': str(value),
                'hash': hashlib.md5(str(value).encode()).hexdigest()[:16]
            }
    
    # 使用JSON序列化来确保一致的输出格式
    dep_str = json.dumps(serializable_deps, sort_keys=True, separators=(',', ':'), default=str)
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