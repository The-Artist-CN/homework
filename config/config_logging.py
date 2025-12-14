# config/config_logging.py
import logging
import logging.config
import os
from datetime import datetime

def setup_logging(debug_mode=False):
    """设置日志配置
    
    Args:
        debug_mode: 是否启用调试模式
    """
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'lending_club_{timestamp}.log')
    
    # 根据模式设置日志级别
    if debug_mode:
        console_level = 'DEBUG'
        file_level = 'DEBUG'
    else:
        console_level = 'INFO'
        file_level = 'DEBUG'  # 文件仍然记录DEBUG，方便调试
    
    # 日志配置字典
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'datefmt': '%H:%M:%S'
            },
            'debug': {
                'format': '%(asctime)s [%(levelname)8s] %(name)s.%(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%H:%M:%S'
            },
            'minimal': {
                'format': '%(message)s'
            }
        },
        
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': console_level,
                'formatter': 'simple' if not debug_mode else 'debug',
                'stream': 'ext://sys.stdout'
            },
            'debug_console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'debug',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': file_level,
                'formatter': 'verbose',
                'filename': log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'debug_file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'verbose',
                'filename': os.path.join(log_dir, f'debug_{timestamp}.log'),
                'encoding': 'utf-8'
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'verbose',
                'filename': os.path.join(log_dir, 'errors.log'),
                'encoding': 'utf-8'
            }
        },
        
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file', 'error_file'],
                'propagate': True
            },
            'data_loader': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'data_cleaner': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'data_explorer': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'feature_engineer': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'data_splitter': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'model_trainer': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'model_evaluator': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'main': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'config': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'debug_file'],
                'propagate': False
            },
            'sklearn': {
                'level': 'WARNING',  # 减少sklearn的详细日志
                'handlers': ['file'],
                'propagate': False
            },
            'matplotlib': {
                'level': 'WARNING',  # 减少matplotlib的详细日志
                'handlers': ['file'],
                'propagate': False
            }
        }
    }
    
    # 应用配置
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # 记录日志初始化信息
    logger = logging.getLogger('config')
    
    if debug_mode:
        logger.info(f"🔧 DEBUG模式已启用")
        logger.info(f"详细日志将输出到控制台")
    else:
        logger.info(f"📋 INFO模式运行")
    
    logger.info(f"日志文件: {log_file}")
    logger.info(f"调试日志: {os.path.join(log_dir, f'debug_{timestamp}.log')}")
    logger.info(f"错误日志: {os.path.join(log_dir, 'errors.log')}")
    
    return log_file

def get_logger(name):
    """获取指定名称的logger"""
    return logging.getLogger(name)

def set_log_level(level_name, debug_mode=False):
    """设置日志级别
    
    Args:
        level_name: 日志级别名称
        debug_mode: 是否调试模式
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(level_name.upper(), logging.DEBUG if debug_mode else logging.INFO)
    
    # 更新所有logger的级别
    loggers_to_update = [
        '', 'main', 'data_loader', 'data_cleaner', 'data_explorer',
        'feature_engineer', 'data_splitter', 'model_trainer', 
        'model_evaluator', 'config'
    ]
    
    for logger_name in loggers_to_update:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    root_logger = logging.getLogger()
    root_logger.info(f"日志级别设置为: {level_name} (DEBUG模式: {debug_mode})")
    
    return level