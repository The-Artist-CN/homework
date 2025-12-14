# src/data_splitter.py
"""
数据分割模块
owner: [D同学学号] + [D同学姓名]
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config.config import Config
from config.config_logging import get_logger

logger = get_logger('data_splitter')

def split_data(df):
    """
    数据分割
    返回: X_train, X_test, y_train, y_test
    """
    logger.info("开始数据分割...")
    
    try:
        # 检查目标变量
        if 'target' not in df.columns:
            logger.error("数据中未找到目标变量 'target'")
            raise ValueError("数据中未找到目标变量 'target'")
        
        # 分离特征和目标
        X = df.drop(columns=['target'])
        y = df['target']
        
        logger.info(f"特征形状: {X.shape}")
        logger.info(f"目标变量形状: {y.shape}")
        logger.info(f"违约率: {y.mean():.3f}")
        
        # 记录类别分布
        class_counts = y.value_counts()
        logger.info(f"类别分布: 0={class_counts.get(0, 0)}, 1={class_counts.get(1, 0)}")
        
        # 分层分割
        logger.info(f"使用分层分割，测试集比例: {Config.TEST_SIZE}")
        logger.info(f"随机种子: {Config.RANDOM_SEED}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            stratify=y if Config.USE_STRATIFIED_SPLIT else None,
            random_state=Config.RANDOM_SEED
        )
        
        # 记录分割结果
        logger.info("数据分割完成:")
        logger.info(f"训练集: {X_train.shape}, 违约率: {y_train.mean():.3f}")
        logger.info(f"测试集: {X_test.shape}, 违约率: {y_test.mean():.3f}")
        
        # 检查特征一致性
        check_feature_consistency(X_train, X_test)
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"数据分割过程中出错: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        raise

def check_feature_consistency(X_train, X_test):
    """检查训练集和测试集特征一致性"""
    logger.debug("检查特征一致性...")
    
    # 检查特征数量
    train_features = set(X_train.columns)
    test_features = set(X_test.columns)
    
    if train_features != test_features:
        missing_in_test = train_features - test_features
        missing_in_train = test_features - train_features
        
        if missing_in_test:
            logger.error(f"测试集缺少特征: {missing_in_test}")
        if missing_in_train:
            logger.error(f"训练集缺少特征: {missing_in_train}")
        
        raise ValueError("训练集和测试集特征不一致")
    else:
        logger.info("✓ 训练集和测试集特征一致")
    
    # 检查缺失值
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    
    if train_missing > 0 or test_missing > 0:
        logger.warning(f"存在缺失值 - 训练集: {train_missing}, 测试集: {test_missing}")
    else:
        logger.info("✓ 无缺失值")
    
    logger.info("特征一致性检查完成")