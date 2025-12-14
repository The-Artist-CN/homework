# src/data_loader.py
"""
数据加载模块
owner: [A同学学号] + [A同学姓名]
"""
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pandas as pd
import numpy as np
from config.config import Config
from config.config_logging import get_logger

logger = get_logger('data_loader')

def load_data():
    """
    加载Lending Club数据
    返回: DataFrame
    """
    logger.info("开始加载数据...")
    
    try:
        # 读取数据
        if not os.path.exists(Config.DATA_PATH):
            logger.error(f"数据文件不存在: {Config.DATA_PATH}")
            logger.info("创建示例数据...")
            df = create_sample_data()
        else:
            logger.info(f"从文件加载数据: {Config.DATA_PATH}")
            df = pd.read_csv(Config.DATA_PATH, low_memory=False)
        
        # 记录基本信息
        logger.info(f"原始数据形状: {df.shape}")
        logger.info(f"原始数据列数: {len(df.columns)}")
        
        # 打印前3行
        logger.info("\n前3行数据:")
        print(df.head(3).to_string())
        
        # 检查关键列是否存在
        missing_cols = [col for col in Config.KEY_COLUMNS if col not in df.columns]
        
        if missing_cols:
            logger.error(f"缺少关键列: {missing_cols}")
            logger.error(f"可用列: {list(df.columns[:20])}...")
            raise ValueError(f"缺少关键列: {missing_cols}")
        else:
            logger.info("✓ 关键列检查通过")
        
        # 记录数据摘要
        generate_data_summary(df)
        
        return df
        
    except Exception as e:
        logger.error(f"数据加载错误: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        raise

def generate_data_summary(df):
    """生成数据摘要"""
    logger.info("生成数据摘要...")
    
    summary = {
        '总记录数': len(df),
        '总特征数': len(df.columns),
        '数值特征数': len(df.select_dtypes(include=[np.number]).columns),
        '类别特征数': len(df.select_dtypes(include=['object']).columns),
        '缺失值总数': df.isnull().sum().sum(),
        '贷款状态分布': df['loan_status'].value_counts().to_dict() if 'loan_status' in df.columns else 'N/A'
    }
    
    # 记录摘要
    logger.info("数据摘要:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    # 保存到文件
    summary_path = os.path.join(Config.REPORT_DIR, 'data_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("数据摘要\n")
        f.write("="*50 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"数据摘要已保存: {summary_path}")
    
    return summary

def create_sample_data():
    """创建示例数据用于测试"""
    logger.info("创建示例数据...")
    
    np.random.seed(Config.RANDOM_SEED)
    n_samples = 1000
    
    # 根据提供的字段创建示例数据
    data = {
        'id': range(1, n_samples + 1),
        'member_id': np.random.randint(100000, 999999, n_samples),
        'loan_amnt': np.random.randint(5000, 35000, n_samples),
        'funded_amnt': np.random.randint(5000, 35000, n_samples),
        'term': np.random.choice(['36 months', '60 months'], n_samples, p=[0.7, 0.3]),
        'int_rate': np.random.uniform(5, 30, n_samples),
        'installment': np.random.uniform(100, 1000, n_samples),
        'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples),
        'sub_grade': [f"{g}{np.random.randint(1,5)}" for g in np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples)],
        'emp_title': np.random.choice(['Teacher', 'Engineer', 'Manager', 'Analyst', 'Sales', 'Nurse', 'Driver', 'Other'], n_samples),
        'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', 
                                       '6 years', '7 years', '8 years', '9 years', '10+ years'], n_samples),
        'home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
        'annual_inc': np.random.randint(30000, 150000, n_samples),
        'verification_status': np.random.choice(['Verified', 'Source Verified', 'Not Verified'], n_samples),
        'issue_d': [f"{np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}-{np.random.choice(['2013', '2014'])}" for _ in range(n_samples)],
        'loan_status': np.random.choice(['Fully Paid', 'Charged Off', 'Current'], n_samples, p=[0.8, 0.15, 0.05]),
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 
                                    'major_purchase', 'small_business', 'car', 'medical'], n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'delinq_2yrs': np.random.randint(0, 5, n_samples),
        'fico_range_low': np.random.randint(600, 800, n_samples),
        'fico_range_high': np.random.randint(650, 850, n_samples),
        'inq_last_6mths': np.random.randint(0, 10, n_samples),
        'open_acc': np.random.randint(1, 30, n_samples),
        'pub_rec': np.random.randint(0, 5, n_samples),
        'revol_bal': np.random.randint(1000, 50000, n_samples),
        'revol_util': np.random.uniform(0, 100, n_samples),
        'total_acc': np.random.randint(5, 60, n_samples),
        'initial_list_status': np.random.choice(['f', 'w'], n_samples),
        'addr_state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值以模拟真实数据
    for col in ['mths_since_last_delinq', 'mths_since_last_record']:
        df[col] = np.random.choice([np.nan] + list(range(1, 60)), n_samples, p=[0.3] + [0.7/59]*59)
    
    logger.info(f"示例数据创建完成: {df.shape}")
    return df