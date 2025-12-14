# config/__init__.py
"""
配置包
提供项目配置参数和日志配置功能
"""

from .config import Config
from .config_logging import setup_logging, get_logger, set_log_level

__version__ = "1.0.0"
__author__ = "Lending Club Project Team"
__all__ = ['Config', 'setup_logging', 'get_logger', 'set_log_level']

# 初始化配置
try:
    Config.init()
except:
    pass  # 防止循环导入

print(f"✓ 配置包已加载 (版本 {__version__})")