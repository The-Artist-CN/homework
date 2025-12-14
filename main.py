# main.py
"""
Lending Club借贷违约风险评估主程序
按照项目分工，每位成员负责自己的函数
"""
import os
import sys
import time
import traceback
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
from config.config_logging import setup_logging, get_logger, set_log_level
from config.config import Config

# 初始化日志
log_file = setup_logging(debug_mode=Config.DEBUG_MODE)
logger = get_logger('main')

# 设置日志级别
set_log_level(Config.LOG_LEVEL, debug_mode=Config.DEBUG_MODE)

# 导入各个模块
try:
    from src.data_loader import load_data
    from src.data_cleaner import clean_data
    from src.data_explorer import explore_data
    from src.feature_engineer import feature_engineering
    from src.data_splitter import split_data
    from src.model_trainer import train_model
    from src.model_evaluator import evaluate_model, save_predictions
    logger.info("✓ 所有模块导入成功")
    
    if Config.DEBUG_MODE:
        logger.debug("已导入模块:")
        logger.debug(f"  data_loader: {load_data.__module__}")
        logger.debug(f"  data_cleaner: {clean_data.__module__}")
        logger.debug(f"  model_trainer: {train_model.__module__}")
        
except ImportError as e:
    logger.error(f"模块导入失败: {e}")
    logger.error("请检查模块文件是否存在或导入语句是否正确")
    logger.error("详细错误:", exc_info=True)
    sys.exit(1)

def setup_project():
    """设置项目目录结构"""
    logger.info("设置项目目录结构...")
    
    if Config.DEBUG_MODE:
        logger.debug("当前工作目录: %s", os.getcwd())
        logger.debug("Python路径: %s", sys.path)
    
    # 创建必要的目录
    directories = [
        'report/figures',
        'outputs/models',
        'data',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"创建/确认目录: {directory}")
    
    # 检查数据文件
    if not os.path.exists(Config.DATA_PATH):
        logger.warning(f"数据文件 {Config.DATA_PATH} 不存在")
        logger.info("将使用示例数据运行...")
    else:
        logger.info(f"找到数据文件: {Config.DATA_PATH}")
        if Config.DEBUG_MODE:
            import pandas as pd
            try:
                df_sample = pd.read_csv(Config.DATA_PATH, nrows=1)
                logger.debug(f"数据文件格式: CSV, 列数: {len(df_sample.columns)}")
            except Exception as e:
                logger.debug(f"无法读取数据文件样本: {e}")
    
    logger.info("项目目录设置完成")

def log_execution_time(func):
    """记录函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"开始执行: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"完成执行: {func.__name__}, 耗时: {execution_time:.2f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"执行失败: {func.__name__}, 耗时: {execution_time:.2f}秒, 错误: {str(e)}")
            logger.error(f"详细错误信息:", exc_info=True)
            raise
    
    return wrapper

def main():
    """主函数 - 项目总流程控制"""
    logger.info("=" * 60)
    logger.info("Lending Club借贷违约风险评估项目")
    logger.info("=" * 60)
    logger.info(f"日志文件: {log_file}")
    
    if Config.DEBUG_MODE:
        logger.debug("Python版本: %s", sys.version)
        logger.debug("工作目录: %s", os.getcwd())
        logger.debug("DEBUG模式: %s", Config.DEBUG_MODE)
    
    try:
        # 验证配置
        Config.validate()
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        sys.exit(1)
    
    # 步骤1: 数据加载
    logger.info("\n[步骤1] 数据加载...")
    try:
        df = load_data()
        logger.info(f"✓ 数据加载完成: {df.shape}")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        sys.exit(1)
    
    # 步骤2: 数据清洗
    logger.info("\n[步骤2] 数据清洗...")
    try:
        df_cleaned = clean_data(df)
        logger.info(f"✓ 数据清洗完成: {df_cleaned.shape}")
    except Exception as e:
        logger.error(f"数据清洗失败: {e}")
        sys.exit(1)
    
    # 步骤3: 数据探索
    logger.info("\n[步骤3] 数据探索...")
    try:
        explore_data(df_cleaned)
        logger.info("✓ 数据探索完成")
    except Exception as e:
        logger.error(f"数据探索失败: {e}")
        # 不退出，继续执行
    
    # 步骤4: 特征工程
    logger.info("\n[步骤4] 特征工程...")
    try:
        df_features = feature_engineering(df_cleaned)
        logger.info(f"✓ 特征工程完成: {df_features.shape}")
    except Exception as e:
        logger.error(f"特征工程失败: {e}")
        sys.exit(1)
    
    # 步骤5: 数据分割
    logger.info("\n[步骤5] 数据分割...")
    try:
        X_train, X_test, y_train, y_test = split_data(df_features)
        logger.info(f"✓ 数据分割完成")
        logger.info(f"  训练集: {X_train.shape}")
        logger.info(f"  测试集: {X_test.shape}")
    except Exception as e:
        logger.error(f"数据分割失败: {e}")
        sys.exit(1)
    
    # 步骤6: 模型训练
    logger.info("\n[步骤6] 模型训练...")
    try:
        models = train_model(X_train, y_train)
        logger.info(f"✓ 模型训练完成，训练了 {len(models)} 个模型")
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        sys.exit(1)
    
    # 步骤7: 模型评估
    logger.info("\n[步骤7] 模型评估...")
    try:
        results = evaluate_model(models, X_test, y_test)
        logger.info("✓ 模型评估完成")
    except Exception as e:
        logger.error(f"模型评估失败: {e}")
        # 不退出，继续执行
    
    # 步骤8: 结果导出
    logger.info("\n[步骤8] 结果导出...")
    try:
        predictions_df = save_predictions(models['logistic_regression'], X_test, y_test)
        logger.info(f"✓ 结果导出完成，保存到: {Config.OUTPUT_PREDICTIONS}")
    except Exception as e:
        logger.error(f"结果导出失败: {e}")
        # 不退出，继续执行
    
    # 总结报告
    logger.info("\n" + "=" * 60)
    logger.info("项目执行总结")
    logger.info("=" * 60)
    
    if 'results' in locals():
        logger.info("模型性能总结:")
        for model_name, metrics in results.items():
            if 'roc_auc' in metrics:
                logger.info(f"  {model_name}: AUC={metrics['roc_auc']:.4f}, F1={metrics.get('f1', 0):.4f}")
    
    if 'predictions_df' in locals():
        correct_predictions = predictions_df['prediction_correct'].sum()
        total_predictions = len(predictions_df)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"预测结果统计:")
        logger.info(f"  总预测数: {total_predictions}")
        logger.info(f"  正确预测数: {correct_predictions}")
        logger.info(f"  预测准确率: {accuracy:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("项目执行完成！")
    logger.info("=" * 60)
    
    return locals().get('results', {})

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        # 设置项目环境
        setup_project()
        
        # 运行主程序
        results = main()
        
        total_time = time.time() - start_time
        logger.info(f"总执行时间: {total_time:.2f}秒")
        logger.info(f"日志文件: {log_file}")
        
        # 退出代码
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("用户中断程序执行")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"项目执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)