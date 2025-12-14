# src/feature_engineer.py
"""
特征工程模块
owner: [C同学学号] + [C同学姓名]
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config.config import Config
from config.config_logging import get_logger

logger = get_logger('feature_engineer')

def feature_engineering(df):
    """
    特征工程：
    1. 创建新特征（至少2个）
    2. 编码类别特征
    3. 特征缩放
    返回: 处理后的特征DataFrame
    """
    logger.info("开始特征工程...")
    initial_shape = df.shape
    logger.info(f"初始数据形状: {initial_shape}")
    
    try:
        df_features = df.copy()
        
        # 1. 创建新特征
        logger.info("1. 创建新特征...")
        df_features = create_new_features(df_features)
        
        # 2. 处理类别特征
        logger.info("2. 处理类别特征...")
        df_features = encode_categorical_features(df_features)
        
        # 3. 特征缩放
        logger.info("3. 特征缩放...")
        df_features = scale_features(df_features)
        
        # 4. 特征选择（删除无用特征）
        logger.info("4. 特征选择...")
        df_features = select_features(df_features)
        
        final_shape = df_features.shape
        logger.info(f"特征工程完成:")
        logger.info(f"  初始形状: {initial_shape}")
        logger.info(f"  最终形状: {final_shape}")
        logger.info(f"  特征数量变化: {initial_shape[1]} -> {final_shape[1]}")
        
        return df_features
        
    except Exception as e:
        logger.error(f"特征工程过程中出错: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        raise

def create_new_features(df):
    """创建新特征（至少2个）"""
    logger.info("创建新特征...")
    df_new = df.copy()
    new_features = []
    
    # 特征1: annual_inc的对数变换（处理偏态分布）
    if Config.FEATURE_ENGINEERING['log_annual_inc'] and 'annual_inc' in df_new.columns:
        df_new['log_annual_inc'] = np.log1p(df_new['annual_inc'])
        new_features.append('log_annual_inc')
        logger.info(f"  创建特征: log_annual_inc")
    
    # 特征2: DTI分箱（债务收入比）
    if Config.FEATURE_ENGINEERING['dti_bins'] and 'dti' in df_new.columns:
        df_new['dti_category'] = pd.cut(df_new['dti'], 
                                       bins=Config.DTI_BINS,
                                       labels=Config.DTI_LABELS)
        new_features.append('dti_category')
        logger.info(f"  创建特征: dti_category")
        
        # 记录分箱分布
        if Config.DEBUG_MODE:
            category_dist = df_new['dti_category'].value_counts()
            logger.debug(f"DTI分箱分布: {category_dist.to_dict()}")
    
    # 特征3: revol_util分段（循环利用率）
    if Config.FEATURE_ENGINEERING['revol_util_groups'] and 'revol_util' in df_new.columns:
        df_new['revol_util_group'] = pd.cut(df_new['revol_util'],
                                          bins=Config.REVOL_UTIL_BINS,
                                          labels=Config.REVOL_UTIL_LABELS)
        new_features.append('revol_util_group')
        logger.info(f"  创建特征: revol_util_group")
    
    # 特征4: loan_amnt与annual_inc的比例
    if Config.FEATURE_ENGINEERING['loan_to_income'] and all(col in df_new.columns for col in ['loan_amnt', 'annual_inc']):
        df_new['loan_to_income_ratio'] = df_new['loan_amnt'] / (df_new['annual_inc'] + 1)
        new_features.append('loan_to_income_ratio')
        logger.info(f"  创建特征: loan_to_income_ratio")
    
    # 特征5: grade编码（字母等级转换为数值）
    if Config.FEATURE_ENGINEERING['grade_numeric'] and 'grade' in df_new.columns:
        grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        df_new['grade_numeric'] = df_new['grade'].map(grade_mapping).fillna(0)
        new_features.append('grade_numeric')
        logger.info(f"  创建特征: grade_numeric")
    
    # 特征6: emp_length转换为数值
    if Config.FEATURE_ENGINEERING['emp_length_numeric'] and 'emp_length' in df_new.columns:
        df_new['emp_length_numeric'] = df_new['emp_length'].apply(convert_emp_length)
        new_features.append('emp_length_numeric')
        logger.info(f"  创建特征: emp_length_numeric")
    
    logger.info(f"共创建了 {len(new_features)} 个新特征")
    return df_new

def convert_emp_length(emp_str):
    """将雇佣时长字符串转换为数值"""
    if pd.isna(emp_str) or emp_str == 'nan':
        return 0
    
    emp_str = str(emp_str).lower()
    
    if '< 1' in emp_str:
        return 0.5
    elif '10+' in emp_str or '10' in emp_str:
        return 10
    else:
        # 提取数字
        numbers = re.findall(r'\d+', emp_str)
        if numbers:
            return float(numbers[0])
        return 0

def encode_categorical_features(df):
    """编码类别特征"""
    logger.info("编码类别特征...")
    df_encoded = df.copy()
    
    # 分离类别特征
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 排除目标变量
    if 'target' in categorical_cols:
        categorical_cols.remove('target')
    
    logger.info(f"需要编码的类别特征数: {len(categorical_cols)}")
    
    encoding_info = []
    
    for col in categorical_cols:
        unique_count = df_encoded[col].nunique()
        
        if unique_count <= 10:  # 类别少：使用One-Hot编码
            logger.debug(f"One-Hot编码: {col} ({unique_count}个类别)")
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            encoding_info.append((col, 'one-hot', unique_count, len(dummies.columns)))
        else:  # 类别多：使用频率编码
            logger.debug(f"频率编码: {col} ({unique_count}个类别)")
            freq = df_encoded[col].value_counts(normalize=True)
            df_encoded[col] = df_encoded[col].map(freq)
            encoding_info.append((col, 'frequency', unique_count, 1))
    
    # 记录编码信息
    logger.info("类别特征编码完成:")
    for col, method, unique_count, new_cols in encoding_info:
        logger.info(f"  {col}: {method}编码, {unique_count}个类别 -> {new_cols}个特征")
    
    return df_encoded

def scale_features(df):
    """特征缩放"""
    if not Config.SCALE_FEATURES:
        logger.info("跳过特征缩放")
        return df
    
    logger.info("特征缩放...")
    df_scaled = df.copy()
    
    # 选择数值特征
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除目标变量和标识列
    exclude_cols = ['target', 'id', 'member_id', 'issue_year']
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if scale_cols:
        logger.info(f"缩放{len(scale_cols)}个特征")
        
        # 记录缩放前的统计信息
        if Config.DEBUG_MODE:
            for col in scale_cols[:5]:  # 只记录前5个
                logger.debug(f"{col}: 均值={df_scaled[col].mean():.2f}, 标准差={df_scaled[col].std():.2f}")
        
        # 使用标准化
        scaler = StandardScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        
        logger.info("特征缩放完成")
    else:
        logger.info("没有需要缩放的特征")
    
    return df_scaled

def select_features(df):
    """特征选择"""
    logger.info("特征选择...")
    df_selected = df.copy()
    
    # 删除无用的标识列
    useless_cols = ['id', 'member_id', 'issue_d', 'issue_year', 'loan_status']
    cols_to_drop = [col for col in useless_cols if col in df_selected.columns]
    
    if cols_to_drop:
        df_selected = df_selected.drop(columns=cols_to_drop)
        logger.info(f"删除无用列: {cols_to_drop}")
    
    # 删除常数列
    constant_cols = []
    for col in df_selected.columns:
        if df_selected[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        df_selected = df_selected.drop(columns=constant_cols)
        logger.info(f"删除常数列: {constant_cols}")
    
    logger.info(f"特征选择后: {len(df_selected.columns)}个特征")
    return df_selected