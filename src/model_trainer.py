# src/model_trainer.py
"""
模型训练模块
owner: [E同学学号] + [E同学姓名]
"""
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# 现在可以安全地导入 config
try:
    from config.config import Config
    from config.config_logging import get_logger
    logger = get_logger('model_trainer')
except ImportError as e:
    # 如果导入失败，创建一个简单的 logger
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('model_trainer')
    logger.warning(f"无法导入配置模块: {e}")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score
except ImportError as e:
    logger.error(f"sklearn 导入失败: {e}")
    raise

def validate_and_convert_data(X_train, y_train):
    """
    验证并转换数据类型，确保模型可以处理
    """
    logger.info("验证和转换数据格式...")
    
    # 1. 记录原始信息
    logger.debug(f"原始数据类型: {X_train.dtypes.tolist()}")
    logger.debug(f"原始数据形状: {X_train.shape}")
    
    # 2. 检查并转换 y_train
    if y_train.dtype not in [np.int64, np.int32, np.float64, np.float32]:
        logger.warning(f"目标变量 y_train 类型为 {y_train.dtype}，转换为 int64")
        y_train = y_train.astype(np.int64)
    
    # 3. 处理 X_train 中的日期时间列
    datetime_cols = X_train.select_dtypes(include=['datetime64', 'datetime64[ns]', 'timedelta', 'timedelta64']).columns
    if len(datetime_cols) > 0:
        logger.warning(f"发现日期时间列: {list(datetime_cols)}")
        
        for col in datetime_cols:
            try:
                # 转换为时间戳（秒数）
                if X_train[col].dtype in ['datetime64[ns]', 'datetime64']:
                    # 方法1: 转换为UNIX时间戳
                    X_train[col] = (X_train[col] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
                    logger.info(f"  转换 {col}: datetime64 -> 时间戳(秒)")
                elif X_train[col].dtype in ['timedelta64[ns]', 'timedelta64']:
                    # 转换为秒数
                    X_train[col] = X_train[col].dt.total_seconds()
                    logger.info(f"  转换 {col}: timedelta64 -> 秒数")
            except Exception as e:
                logger.error(f"转换列 {col} 失败: {e}")
                # 如果转换失败，尝试其他方法
                try:
                    X_train[col] = pd.to_numeric(X_train[col])
                    logger.info(f"  转换 {col}: 强制转换为数值")
                except:
                    logger.error(f"  无法转换列 {col}，将删除该列")
                    X_train = X_train.drop(col, axis=1)
    
    # 4. 处理对象/字符串列
    object_cols = X_train.select_dtypes(include=['object', 'category']).columns
    if len(object_cols) > 0:
        logger.warning(f"发现对象类型列: {list(object_cols)}")
        
        for col in object_cols:
            try:
                # 尝试转换为数值
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                # 填充NaN
                X_train[col] = X_train[col].fillna(0)
                logger.info(f"  转换 {col}: object -> 数值")
            except:
                # 如果是分类变量，使用独热编码或标签编码
                n_unique = X_train[col].nunique()
                if n_unique < 10:
                    # 类别较少，使用独热编码
                    dummies = pd.get_dummies(X_train[col], prefix=col, drop_first=True)
                    X_train = pd.concat([X_train.drop(col, axis=1), dummies], axis=1)
                    logger.info(f"  转换 {col}: object -> 独热编码 ({n_unique}个类别)")
                else:
                    # 类别较多，使用标签编码
                    X_train[col] = pd.factorize(X_train[col])[0]
                    logger.info(f"  转换 {col}: object -> 标签编码 ({n_unique}个类别)")
    
    # 5. 处理布尔类型
    bool_cols = X_train.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        logger.info(f"发现布尔类型列: {list(bool_cols)}")
        for col in bool_cols:
            X_train[col] = X_train[col].astype(np.int8)
            logger.info(f"  转换 {col}: bool -> int8")
    
    # 6. 确保所有列都是数值类型
    for col in X_train.columns:
        if X_train[col].dtype not in [np.int64, np.int32, np.float64, np.float32, np.int8, np.int16]:
            try:
                X_train[col] = X_train[col].astype(np.float64)
            except:
                logger.error(f"无法转换列 {col} 为数值类型，将删除")
                X_train = X_train.drop(col, axis=1)
    
    # 7. 处理可能的NaN值
    if X_train.isnull().any().any():
        logger.warning(f"发现NaN值，填充为0")
        X_train = X_train.fillna(0)
    
    logger.info(f"转换后数据类型: {X_train.dtypes.tolist()}")
    logger.info(f"转换后数据形状: {X_train.shape}")
    
    return X_train, y_train

def train_model(X_train, y_train):
    """
    训练多个模型
    返回: 训练好的模型字典
    """
    logger.info("开始模型训练...")
    
    # 首先验证和转换数据
    logger.info("数据预处理阶段...")
    X_train_processed, y_train_processed = validate_and_convert_data(X_train.copy(), y_train.copy())
    
    logger.info(f"训练数据形状: {X_train_processed.shape}")
    logger.info(f"目标变量分布: 0={sum(y_train_processed==0)}, 1={sum(y_train_processed==1)}")
    
    models = {}
    
    try:
        # 1. 逻辑回归（基线模型）
        logger.info("\n1. 训练逻辑回归模型...")
        lr_model = train_logistic_regression(X_train_processed, y_train_processed)
        models['logistic_regression'] = lr_model
        
        # 2. 随机森林
        logger.info("\n2. 训练随机森林模型...")
        rf_model = train_random_forest(X_train_processed, y_train_processed)
        models['random_forest'] = rf_model
        
        # 3. 决策树（可选）
        logger.info("\n3. 训练决策树模型...")
        dt_model = train_decision_tree(X_train_processed, y_train_processed)
        models['decision_tree'] = dt_model
        
        # 保存模型
        save_models(models)
        
        logger.info(f"✓ 模型训练完成，共训练{len(models)}个模型")
        return models
        
    except Exception as e:
        logger.error(f"模型训练过程中出错: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        raise

def train_logistic_regression(X_train, y_train):
    """训练逻辑回归模型"""
    start_time = time.time()
    logger.info("训练逻辑回归模型...")
    
    try:
        # 添加调试信息
        logger.debug(f"输入数据信息:")
        logger.debug(f"  X_train 数据类型: {X_train.dtypes.tolist()}")
        logger.debug(f"  X_train 形状: {X_train.shape}")
        logger.debug(f"  y_train 数据类型: {y_train.dtype}")
        
        # 确保数据是数值类型
        X_train_np = np.array(X_train, dtype=np.float64)
        y_train_np = np.array(y_train, dtype=np.int64)
        
        # 获取配置参数
        lr_params = Config.MODELS['logistic_regression']
        logger.debug(f"逻辑回归参数: {lr_params}")
        
        # 创建模型
        lr = LogisticRegression(
            C=lr_params['C'],
            class_weight=lr_params['class_weight'],
            max_iter=lr_params['max_iter'],
            random_state=lr_params['random_state'],
            solver='liblinear'  # 适合小到中型数据集
        )
        
        logger.info("开始训练...")
        lr.fit(X_train_np, y_train_np)
        logger.info("训练完成")
        
        # 交叉验证
        logger.info("进行交叉验证...")
        cv_scores = cross_val_score(lr, X_train_np, y_train_np, cv=5, scoring='roc_auc')
        
        training_time = time.time() - start_time
        
        logger.info(f"逻辑回归训练完成:")
        logger.info(f"  耗时: {training_time:.2f}秒")
        logger.info(f"  交叉验证AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        logger.info(f"  交叉验证AUC详情: {cv_scores}")
        
        return lr
        
    except Exception as e:
        logger.error(f"逻辑回归训练失败: {str(e)}")
        # 提供更多调试信息
        logger.error(f"数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.error(f"X_train 数据类型: {X_train.dtypes.tolist() if hasattr(X_train, 'dtypes') else 'Not DataFrame'}")
        raise

# 其他函数保持不变
def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    start_time = time.time()
    logger.info("训练随机森林模型...")
    
    try:
        # 获取配置参数
        rf_params = Config.MODELS['random_forest']
        logger.debug(f"随机森林参数: {rf_params}")
        
        # 创建模型
        rf = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            class_weight=rf_params['class_weight'],
            random_state=rf_params['random_state'],
            n_jobs=-1  # 使用所有CPU核心
        )
        
        logger.info("开始训练...")
        rf.fit(X_train, y_train)
        logger.info("训练完成")
        
        # 交叉验证（使用较少折数以节省时间）
        logger.info("进行交叉验证...")
        cv_scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
        
        training_time = time.time() - start_time
        
        logger.info(f"随机森林训练完成:")
        logger.info(f"  耗时: {training_time:.2f}秒")
        logger.info(f"  交叉验证AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        logger.info(f"  交叉验证AUC详情: {cv_scores}")
        
        return rf
        
    except Exception as e:
        logger.error(f"随机森林训练失败: {str(e)}")
        raise

def train_decision_tree(X_train, y_train):
    """训练决策树模型"""
    start_time = time.time()
    logger.info("训练决策树模型...")
    
    try:
        # 创建模型
        dt = DecisionTreeClassifier(
            max_depth=8,
            class_weight='balanced',
            random_state=Config.RANDOM_SEED
        )
        
        logger.debug(f"决策树参数: max_depth=8, class_weight='balanced'")
        
        logger.info("开始训练...")
        dt.fit(X_train, y_train)
        logger.info("训练完成")
        
        training_time = time.time() - start_time
        
        logger.info(f"决策树训练完成:")
        logger.info(f"  耗时: {training_time:.2f}秒")
        
        return dt
        
    except Exception as e:
        logger.error(f"决策树训练失败: {str(e)}")
        raise

def save_models(models):
    """保存训练好的模型"""
    import os
    
    try:
        # 创建输出目录
        os.makedirs(Config.OUTPUT_MODELS_DIR, exist_ok=True)
        
        logger.info("保存模型...")
        
        for name, model in models.items():
            filename = os.path.join(Config.OUTPUT_MODELS_DIR, f"{name}.pkl")
            
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"  已保存模型: {filename}")
        
        logger.info(f"模型保存完成，共保存{len(models)}个模型")
        
    except Exception as e:
        logger.error(f"模型保存失败: {str(e)}")
        raise