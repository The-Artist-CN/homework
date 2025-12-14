# src/data_cleaner.py
"""
数据清洗模块
owner: [A同学学号] + [A同学姓名]
"""
import pandas as pd
import numpy as np
import re
import os
from config.config import Config
from config.config_logging import get_logger

logger = get_logger('data_cleaner')

def clean_data(df):
    """
    清洗数据：
    1. 筛选2013-2014年、36个月的贷款
    2. 映射目标变量
    3. 剔除贷后字段
    4. 处理缺失值和异常值
    返回: 清洗后的DataFrame
    """
    logger.info("开始数据清洗...")
    initial_shape = df.shape
    logger.info(f"初始数据形状: {initial_shape}")
    
    try:
        df_clean = df.copy()
        
        # 1. 筛选指定年份和期限的贷款
        logger.info("1. 筛选2013-2014年、36个月的贷款...")
        df_clean = filter_by_year_and_term(df_clean)
        
        # 检查筛选后是否有数据
        if len(df_clean) == 0:
            logger.error("筛选后无数据！请检查数据文件是否符合筛选条件")
            logger.error(f"筛选条件: 年份{Config.ISSUE_YEAR_START}-{Config.ISSUE_YEAR_END}, 期限{Config.TERM}")
            raise ValueError("筛选后无数据")
        
        # 2. 映射目标变量
        logger.info("2. 映射目标变量...")
        df_clean = map_target_variable(df_clean)
        
        # 3. 剔除贷后字段
        logger.info("3. 剔除贷后字段...")
        df_clean, removed_cols = remove_post_loan_features(df_clean)
        
        # 4. 处理缺失值
        logger.info("4. 处理缺失值...")
        df_clean = handle_missing_values(df_clean)
        
        # 5. 处理异常值
        logger.info("5. 处理异常值...")
        df_clean = handle_outliers(df_clean)
        
        # 6. 统一类别格式
        logger.info("6. 统一类别格式...")
        df_clean = standardize_categories(df_clean)
        
        final_shape = df_clean.shape
        logger.info(f"清洗完成:")
        logger.info(f"  初始形状: {initial_shape}")
        logger.info(f"  最终形状: {final_shape}")
        logger.info(f"  剔除字段数: {len(removed_cols)}")
        logger.info(f"  保留字段数: {len(df_clean.columns)}")
        
        # 记录剔除的字段
        log_removed_columns(removed_cols)
        
        return df_clean
        
    except Exception as e:
        logger.error(f"数据清洗过程中出错: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        raise

def filter_by_year_and_term(df):
    """筛选2013-2014年、36个月的贷款"""
    initial_count = len(df)
    
    logger.info(f"原始issue_d前5个值: {df['issue_d'].head().tolist()}")
    logger.info(f"term分布: {df['term'].value_counts().to_dict()}")
    
    # 提取年份 - 支持多种日期格式
    try:
        # 首先尝试解析为日期
        df['issue_date'] = pd.to_datetime(df['issue_d'], errors='coerce')
        
        # 检查解析结果
        valid_dates = df['issue_date'].notna().sum()
        logger.info(f"成功解析的日期数量: {valid_dates}/{initial_count}")
        
        if valid_dates == 0:
            logger.warning("无法自动解析日期，尝试手动处理...")
            # 尝试手动提取年份
            df['issue_year'] = extract_year_from_string(df['issue_d'])
        else:
            # 从日期中提取年份
            df['issue_year'] = df['issue_date'].dt.year
            
    except Exception as e:
        logger.warning(f"日期解析出错: {e}")
        logger.warning("尝试手动提取年份...")
        df['issue_year'] = extract_year_from_string(df['issue_d'])
    
    # 检查年份提取结果
    valid_years = df['issue_year'].notna().sum()
    logger.info(f"有效年份数据: {valid_years}/{initial_count}")
    
    if valid_years > 0:
        # 显示年份分布
        year_dist = df['issue_year'].value_counts().sort_index()
        logger.info(f"年份分布: {year_dist.to_dict()}")
    else:
        logger.error("无法提取任何有效年份！")
        # 显示一些样本用于调试
        for i in range(min(5, len(df))):
            logger.error(f"  样本{i}: issue_d='{df['issue_d'].iloc[i]}' -> issue_year={df['issue_year'].iloc[i]}")
    
    # 筛选条件
    mask = (
        (df['issue_year'].between(Config.ISSUE_YEAR_START, Config.ISSUE_YEAR_END)) &
        (df['term'] == Config.TERM)
    )
    
    filtered_df = df[mask].copy()
    
    logger.info(f"筛选结果:")
    logger.info(f"  初始记录数: {initial_count}")
    logger.info(f"  筛选后记录数: {len(filtered_df)}")
    logger.info(f"  删除记录数: {initial_count - len(filtered_df)}")
    
    # 如果没有数据，记录详细信息用于调试
    if len(filtered_df) == 0:
        logger.warning("筛选后无数据，显示数据样本用于调试:")
        logger.warning(f"  issue_d前5个值: {df['issue_d'].head().tolist()}")
        logger.warning(f"  issue_year前5个值: {df['issue_year'].head().tolist()}")
        logger.warning(f"  term分布: {df['term'].value_counts().to_dict()}")
        logger.warning(f"  年份范围: {Config.ISSUE_YEAR_START}-{Config.ISSUE_YEAR_END}")
    
    return filtered_df

def extract_year_from_string(date_series):
    """从字符串中提取年份"""
    years = []
    
    for date_str in date_series:
        if pd.isna(date_str):
            years.append(np.nan)
            continue
            
        date_str = str(date_str)
        
        # 尝试多种模式匹配年份
        try:
            # 模式1: YYYY-MM-DD
            match = re.search(r'(\d{4})-\d{2}-\d{2}', date_str)
            if match:
                years.append(int(match.group(1)))
                continue
            
            # 模式2: MM/DD/YYYY
            match = re.search(r'\d{2}/\d{2}/(\d{4})', date_str)
            if match:
                years.append(int(match.group(1)))
                continue
            
            # 模式3: DD-MMM-YYYY 或 MMM-YYYY
            match = re.search(r'(\d{4})', date_str)
            if match:
                years.append(int(match.group(1)))
                continue
            
            # 模式4: 直接包含4位数字
            matches = re.findall(r'\d{4}', date_str)
            if matches:
                years.append(int(matches[0]))
                continue
            
            years.append(np.nan)
            
        except Exception as e:
            logger.debug(f"提取年份失败: {date_str}, 错误: {e}")
            years.append(np.nan)
    
    return pd.Series(years, index=date_series.index)

def map_target_variable(df):
    """映射目标变量"""
    initial_count = len(df)
    
    if initial_count == 0:
        logger.error("传入的数据为空，无法映射目标变量")
        raise ValueError("数据为空")
    
    # 记录原始分布
    status_dist = df['loan_status'].value_counts()
    logger.info(f"原始loan_status分布:")
    for status, count in status_dist.items():
        percentage = (count/initial_count*100) if initial_count > 0 else 0
        logger.info(f"  {status}: {count} ({percentage:.1f}%)")
    
    # 映射loan_status
    df['target'] = df['loan_status'].map(Config.TARGET_MAPPING)
    
    # 检查映射结果
    mapping_counts = df['target'].value_counts(dropna=False)
    logger.info(f"映射结果分布:")
    for value, count in mapping_counts.items():
        logger.info(f"  {value}: {count}条记录")
    
    # 显示无法映射的loan_status值
    unmapped = df[df['target'].isna()]['loan_status'].unique()
    if len(unmapped) > 0:
        logger.info(f"无法映射的loan_status值 ({len(unmapped)}个):")
        for status in unmapped[:10]:  # 只显示前10个
            logger.info(f"  '{status}'")
        if len(unmapped) > 10:
            logger.info(f"  还有 {len(unmapped)-10} 个...")
    
    # 删除无法映射的记录
    df_mapped = df[df['target'].notna()].copy()
    removed_count = len(df) - len(df_mapped)
    
    if len(df_mapped) == 0:
        logger.error("映射后无有效数据！所有记录都被排除")
        logger.error("可能的loan_status值:")
        for status in df['loan_status'].unique():
            logger.error(f"  '{status}' -> 映射为: {Config.TARGET_MAPPING.get(status, '未定义')}")
        raise ValueError("映射后无有效数据")
    
    # 转换类型
    df_mapped['target'] = df_mapped['target'].astype(int)
    
    # 记录映射后分布
    target_dist = df_mapped['target'].value_counts()
    logger.info(f"目标变量映射完成:")
    logger.info(f"  初始记录数: {initial_count}")
    logger.info(f"  删除无法映射记录: {removed_count}条")
    logger.info(f"  最终记录数: {len(df_mapped)}")
    
    # 安全计算百分比
    total_records = len(df_mapped)
    if total_records > 0:
        for target_value in [0, 1]:
            count = target_dist.get(target_value, 0)
            percentage = (count/total_records*100)
            status = "非违约" if target_value == 0 else "违约"
            logger.info(f"  {status}({target_value}): {count}条 ({percentage:.1f}%)")
    else:
        logger.warning("最终记录数为0，无法计算百分比")
    
    return df_mapped

def remove_post_loan_features(df):
    """剔除贷后字段"""
    initial_cols = set(df.columns)
    
    # 识别贷后字段
    post_loan_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in Config.POST_LOAN_KEYWORDS):
            post_loan_cols.append(col)
    
    # 只保留放款前字段
    pre_loan_cols = [col for col in df.columns if col in Config.PRE_LOAN_COLUMNS or col in ['target', 'issue_year', 'issue_date']]
    
    # 合并要保留的列
    cols_to_keep = list(set(pre_loan_cols) - set(post_loan_cols))
    df_clean = df[cols_to_keep]
    
    # 记录剔除的字段
    removed_cols = list(initial_cols - set(df_clean.columns))
    
    return df_clean, removed_cols

def log_removed_columns(removed_cols):
    """记录剔除的字段"""
    if removed_cols:
        logger.info(f"剔除的贷后字段 ({len(removed_cols)}个):")
        removed_file = os.path.join(Config.REPORT_DIR, 'removed_columns.txt')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(removed_file), exist_ok=True)
        
        with open(removed_file, 'w', encoding='utf-8') as f:
            f.write("剔除的贷后字段清单\n")
            f.write("="*50 + "\n\n")
            f.write("剔除原则: 只使用放款前可获得信息进行建模\n")
            f.write(f"剔除关键词: {Config.POST_LOAN_KEYWORDS}\n\n")
            f.write(f"共剔除 {len(removed_cols)} 个字段:\n\n")
            
            for i, col in enumerate(sorted(removed_cols), 1):
                f.write(f"{i:3d}. {col}\n")
        
        # 在日志中显示前10个
        for col in sorted(removed_cols)[:10]:
            logger.info(f"  - {col}")
        if len(removed_cols) > 10:
            logger.info(f"  ... 还有{len(removed_cols)-10}个字段")
        
        logger.info(f"完整清单已保存: {removed_file}")

def handle_missing_values(df):
    """处理缺失值"""
    logger.info("处理缺失值...")
    
    df_clean = df.copy()
    
    # 1. 删除缺失比例过高的列
    missing_ratio = df_clean.isnull().sum() / len(df_clean)
    high_missing_cols = missing_ratio[missing_ratio > Config.MISSING_THRESHOLD].index.tolist()
    
    if high_missing_cols:
        logger.info(f"删除缺失率>{Config.MISSING_THRESHOLD:.0%}的列 ({len(high_missing_cols)}个)")
        df_clean = df_clean.drop(columns=high_missing_cols)
    
    # 2. 数值列用中位数填充
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            if Config.DEBUG_MODE:
                logger.debug(f"数值列填充: {col} -> 中位数 {median_val:.2f}")
    
    # 3. 类别列用众数填充
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            if not df_clean[col].mode().empty:
                mode_val = df_clean[col].mode()[0]
            else:
                mode_val = 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)
            if Config.DEBUG_MODE:
                logger.debug(f"类别列填充: {col} -> 众数 '{mode_val}'")
    
    remaining_missing = df_clean.isnull().sum().sum()
    logger.info(f"缺失值处理完成，剩余缺失值: {remaining_missing}")
    
    if remaining_missing > 0 and Config.DEBUG_MODE:
        # 显示还有哪些列有缺失值
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        logger.debug(f"仍有缺失值的列: {missing_cols}")
    
    return df_clean

def handle_outliers(df):
    """处理异常值"""
    logger.info("处理异常值...")
    
    df_clean = df.copy()
    
    # 只处理数值列
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # 排除目标变量
    if 'target' in numeric_cols:
        numeric_cols = numeric_cols.drop('target')
    
    outlier_counts = {}
    
    for col in numeric_cols:
        if df_clean[col].nunique() > 10:  # 只对有足够多样本的处理
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # 避免除零
                lower_bound = Q1 - Config.OUTLIER_IQR_MULTIPLIER * IQR
                upper_bound = Q3 + Config.OUTLIER_IQR_MULTIPLIER * IQR
                
                # 检测异常值数量
                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    # 将异常值截断到边界
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    outlier_counts[col] = outlier_count
    
    if outlier_counts:
        logger.info(f"处理了 {len(outlier_counts)} 个特征的异常值:")
        for col, count in outlier_counts.items():
            percentage = (count/len(df_clean)*100) if len(df_clean) > 0 else 0
            logger.info(f"  {col}: {count} 个异常值 ({percentage:.1f}%)")
    
    return df_clean

def standardize_categories(df):
    """统一类别格式"""
    logger.info("统一类别格式...")
    
    df_clean = df.copy()
    
    # 统一文本格式：去除首尾空格
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    
    logger.info("类别格式统一完成")
    return df_clean