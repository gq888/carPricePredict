
import pickle
import hashlib
import os
from typing import Dict, Any, Tuple

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