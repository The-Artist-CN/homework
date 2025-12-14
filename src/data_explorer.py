# src/data_explorer.py
"""
数据探索模块
owner: [B同学学号] + [B同学姓名]
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config.config import Config
from config.config_logging import get_logger

logger = get_logger('data_explorer')

def explore_data(df):
    """
    探索性数据分析
    生成关键统计信息和可视化
    """
    logger.info("开始探索性数据分析...")
    
    try:
        # 确保目录存在
        os.makedirs('report/figures', exist_ok=True)
        os.makedirs('report', exist_ok=True)
        
        logger.info("="*60)
        logger.info("探索性数据分析")
        logger.info("="*60)
        
        # 1. 基本统计信息
        logger.info("1. 基本统计信息")
        print_basic_stats(df)
        
        # 2. 目标变量分析
        logger.info("2. 目标变量分析")
        analyze_target(df)
        
        # 3. 关键特征分布
        logger.info("3. 关键特征分布")
        plot_key_distributions(df)
        
        # 4. 特征相关性分析
        logger.info("4. 特征相关性分析")
        plot_correlation(df)
        
        # 5. 生成数据摘要
        logger.info("5. 生成数据摘要")
        generate_data_summary(df)
        
        logger.info("✓ 数据探索完成")
        
    except Exception as e:
        logger.error(f"数据探索过程中出错: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        raise

def print_basic_stats(df):
    """打印基本统计信息"""
    try:
        # 数据基本信息
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"特征数量: {len(df.columns)}")
        
        # 数据类型分布
        dtype_counts = df.dtypes.value_counts()
        logger.info("数据类型分布:")
        for dtype, count in dtype_counts.items():
            logger.info(f"  {dtype}: {count}")
        
        # 数值特征描述统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.info("数值特征描述统计:")
            stats_df = df[numeric_cols].describe().round(2)
            
            # 记录每个数值特征的统计信息
            for col in numeric_cols:
                stats = stats_df[col]
                logger.debug(f"{col}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, "
                           f"最小值={stats['min']:.2f}, 最大值={stats['max']:.2f}")
        
        # 类别特征唯一值数量
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            logger.info("类别特征唯一值数量:")
            for col in list(categorical_cols)[:10]:  # 只显示前10个
                unique_count = df[col].nunique()
                logger.info(f"  {col}: {unique_count}个唯一值")
                logger.debug(f"{col}的前5个值: {df[col].unique()[:5]}")
            
            if len(categorical_cols) > 10:
                logger.info(f"  还有{len(categorical_cols)-10}个类别特征未显示")
        
    except Exception as e:
        logger.error(f"基本统计信息生成出错: {str(e)}")
        raise

def analyze_target(df):
    """分析目标变量"""
    try:
        if 'target' not in df.columns:
            logger.warning("未找到目标变量")
            return
        
        # 分布情况
        target_dist = df['target'].value_counts()
        target_pct = df['target'].value_counts(normalize=True) * 100
        
        logger.info("目标变量分布:")
        logger.info(f"  违约(1): {target_dist.get(1, 0)}条 ({target_pct.get(1, 0):.1f}%)")
        logger.info(f"  非违约(0): {target_dist.get(0, 0)}条 ({target_pct.get(0, 0):.1f}%)")
        
        # 可视化
        logger.info("生成目标变量分析图表...")
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        target_dist.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('目标变量分布')
        plt.xlabel('是否违约')
        plt.ylabel('数量')
        plt.xticks(rotation=0)
        
        # 违约率随时间变化
        if 'issue_year' in df.columns:
            plt.subplot(1, 2, 2)
            default_by_year = df.groupby('issue_year')['target'].mean() * 100
            default_by_year.plot(kind='line', marker='o')
            plt.title('违约率随时间变化')
            plt.xlabel('发放年份')
            plt.ylabel('违约率(%)')
            plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        chart_path = 'report/figures/target_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图表已保存: {chart_path}")
        
    except Exception as e:
        logger.error(f"目标变量分析出错: {str(e)}")
        raise

def plot_key_distributions(df):
    """绘制关键特征分布"""
    try:
        logger.info("绘制关键特征分布...")
        
        # 选择关键特征
        key_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util']
        available_features = [f for f in key_features if f in df.columns]
        
        if not available_features:
            logger.warning("未找到关键特征")
            return
        
        logger.info(f"绘制以下特征分布: {available_features}")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:6]):  # 最多显示6个
            ax = axes[i]
            
            if df[feature].dtype in [np.float64, np.int64]:
                # 数值特征：直方图
                ax.hist(df[feature].dropna(), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                ax.set_title(f'{feature} 分布')
                ax.set_xlabel(feature)
                ax.set_ylabel('频数')
                
                # 添加统计信息
                stats_text = f"均值: {df[feature].mean():.2f}\n中位数: {df[feature].median():.2f}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                logger.debug(f"{feature}: 均值={df[feature].mean():.2f}, "
                           f"中位数={df[feature].median():.2f}, "
                           f"标准差={df[feature].std():.2f}")
            else:
                # 类别特征：条形图（显示前10个类别）
                top_categories = df[feature].value_counts().head(10)
                top_categories.plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title(f'{feature} (Top 10)')
                ax.set_xlabel(feature)
                ax.set_ylabel('数量')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                logger.debug(f"{feature} Top 3: {top_categories.head(3).to_dict()}")
        
        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # 保存图表
        plt.tight_layout()
        chart_path = 'report/figures/feature_distributions.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"特征分布图表已保存: {chart_path}")
        
    except Exception as e:
        logger.error(f"特征分布绘图出错: {str(e)}")
        raise

def plot_correlation(df):
    """绘制特征相关性热图"""
    try:
        logger.info("进行特征相关性分析...")
        
        # 选择数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logger.warning("数值特征不足，跳过相关性分析")
            return
        
        logger.info(f"分析{len(numeric_cols)}个数值特征的相关性")
        
        # 如果特征太多，只选择部分
        if len(numeric_cols) > 15:
            # 选择与目标变量相关性最高的特征
            if 'target' in df.columns:
                corr_with_target = df[numeric_cols].corrwith(df['target']).abs().sort_values(ascending=False)
                top_features = corr_with_target.head(15).index.tolist()
                numeric_cols = top_features
                logger.info(f"选择与目标变量相关性最高的{len(numeric_cols)}个特征")
        
        # 计算相关系数矩阵
        corr_matrix = df[numeric_cols].corr()
        
        # 绘制热图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('特征相关性热图', fontsize=16)
        
        # 保存图表
        plt.tight_layout()
        chart_path = 'report/figures/correlation_heatmap.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"相关性热图已保存: {chart_path}")
        
        # 打印与目标变量相关性最高的特征
        if 'target' in df.columns:
            logger.info("与目标变量相关性最高的特征:")
            target_corr = df[numeric_cols].corrwith(df['target']).abs().sort_values(ascending=False)
            
            for feature, corr_value in target_corr.head(10).items():
                logger.info(f"  {feature}: {corr_value:.3f}")
        
    except Exception as e:
        logger.error(f"相关性分析出错: {str(e)}")
        raise

def generate_data_summary(df):
    """生成数据摘要"""
    try:
        logger.info("生成数据摘要...")
        
        summary = {
            '总记录数': len(df),
            '总特征数': len(df.columns),
            '数值特征数': len(df.select_dtypes(include=[np.number]).columns),
            '类别特征数': len(df.select_dtypes(include=['object']).columns),
            '缺失值总数': df.isnull().sum().sum(),
            '违约样本数': df['target'].sum() if 'target' in df.columns else 'N/A',
            '非违约样本数': (1 - df['target']).sum() if 'target' in df.columns else 'N/A',
            '违约率': f"{df['target'].mean()*100:.2f}%" if 'target' in df.columns else 'N/A'
        }
        
        # 记录摘要信息
        logger.info("数据摘要:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # 保存摘要到文件
        summary_path = 'report/data_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("数据摘要\n")
            f.write("="*50 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"数据摘要已保存: {summary_path}")
        
        return summary
        
    except Exception as e:
        logger.error(f"生成数据摘要出错: {str(e)}")
        raise