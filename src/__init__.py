# src/__init__.py
"""
源代码包
Lending Club借贷违约风险评估的核心功能模块
"""

__version__ = "1.0.0"
__author__ = "Lending Club Project Team"

# 按模块导入主要功能
try:
    from .data_loader import load_data
    from .data_cleaner import clean_data
    from .data_explorer import explore_data
    from .feature_engineer import feature_engineering
    from .data_splitter import split_data
    from .model_trainer import train_model
    from .model_evaluator import evaluate_model, save_predictions
    
    __all__ = [
        'load_data',
        'clean_data', 
        'explore_data',
        'feature_engineering',
        'split_data',
        'train_model',
        'evaluate_model',
        'save_predictions'
    ]
    
    print(f"✓ 源代码包已加载 (版本 {__version__})")
    print("可用函数:", ', '.join(__all__))
    
except ImportError as e:
    print(f"⚠ 部分模块导入失败: {e}")
    __all__ = []