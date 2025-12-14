# src/model_evaluator.py
"""
模型评估模块
owner: [F同学学号] + [F同学姓名]
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
import joblib
from config.config import Config
from config.config_logging import get_logger

logger = get_logger('model_evaluator')

def evaluate_model(models, X_test, y_test):
    """
    评估模型性能
    返回: 评估结果字典
    """
    logger.info("开始模型评估...")
    logger.info(f"测试数据形状: {X_test.shape}")
    logger.info(f"测试集违约率: {y_test.mean():.3f}")
    
    results = {}
    
    try:
        for name, model in models.items():
            logger.info(f"\n评估模型: {name}")
            model_results = evaluate_single_model(model, X_test, y_test, name)
            results[name] = model_results
        
        # 比较所有模型
        compare_models(results)
        
        # 可视化结果
        plot_model_comparison(results)
        plot_roc_curves(models, X_test, y_test)
        plot_feature_importance(models['logistic_regression'], X_test.columns)
        
        logger.info("✓ 模型评估完成")
        return results
        
    except Exception as e:
        logger.error(f"模型评估过程中出错: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        raise

def evaluate_single_model(model, X_test, y_test, model_name):
    """评估单个模型"""
    start_time = time.time()
    logger.info(f"评估模型{model_name}...")
    
    try:
        # ====== 新增：数据预处理，确保数据类型兼容 ======
        X_test_processed = ensure_numeric_data(X_test.copy())
        
        # 检查并处理NaN值
        if X_test_processed.isnull().any().any():
            logger.warning("测试数据中存在NaN值，将进行填充")
            X_test_processed = X_test_processed.fillna(0)
        
        logger.debug(f"处理后的数据类型: {X_test_processed.dtypes.unique()}")
        logger.debug(f"处理后的数据形状: {X_test_processed.shape}")
        # ==============================================
        
        # 预测
        logger.debug("进行预测...")
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        evaluation_time = time.time() - start_time
        
        # 记录结果
        logger.info(f"{model_name}性能指标:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  评估耗时: {evaluation_time:.2f}秒")
        
        # 分类报告
        logger.debug("分类报告:")
        report = classification_report(y_test, y_pred, target_names=['非违约', '违约'], output_dict=True)
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                logger.debug(f"  {class_name}: precision={class_metrics['precision']:.3f}, "
                           f"recall={class_metrics['recall']:.3f}, f1={class_metrics['f1-score']:.3f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        logger.debug(f"混淆矩阵:\n{cm}")
        
        # 计算额外指标
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'evaluation_time': evaluation_time
        })
        
        logger.debug(f"详细指标:")
        logger.debug(f"  True Negative: {tn}")
        logger.debug(f"  False Positive: {fp}")
        logger.debug(f"  False Negative: {fn}")
        logger.debug(f"  True Positive: {tp}")
        logger.debug(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        logger.debug(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        
        # 保存预测概率（用于后续分析）
        metrics['y_pred_proba'] = y_pred_proba
        
        logger.info(f"✓ 模型{model_name}评估完成")
        return metrics
        
    except Exception as e:
        logger.error(f"模型{model_name}评估失败: {str(e)}")
        raise

def compare_models(results):
    """比较不同模型的性能"""
    logger.info("\n比较模型性能...")
    
    try:
        # 创建比较表格
        comparison_data = []
        
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'FPR': f"{metrics['false_positive_rate']:.4f}",
                'FNR': f"{metrics['false_negative_rate']:.4f}",
                'Time(s)': f"{metrics['evaluation_time']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info("\n模型性能比较:")
        logger.info("\n" + comparison_df.to_string(index=False))
        
        # 找出最佳模型
        best_by_auc = max(results.items(), key=lambda x: x[1]['roc_auc'])
        best_by_f1 = max(results.items(), key=lambda x: x[1]['f1'])
        best_by_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        
        logger.info(f"\n最佳模型:")
        logger.info(f"  按AUC: {best_by_auc[0]} (AUC = {best_by_auc[1]['roc_auc']:.4f})")
        logger.info(f"  按F1: {best_by_f1[0]} (F1 = {best_by_f1[1]['f1']:.4f})")
        logger.info(f"  按Accuracy: {best_by_accuracy[0]} (Accuracy = {best_by_accuracy[1]['accuracy']:.4f})")
        
        # 保存比较结果
        comparison_path = 'outputs/model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"比较结果已保存至: {comparison_path}")
        
        return comparison_df
        
    except Exception as e:
        logger.error(f"模型比较失败: {str(e)}")
        raise

def plot_model_comparison(results):
    """绘制模型比较图"""
    logger.info("绘制模型比较图...")
    
    try:
        # 确保目录存在
        os.makedirs('report/figures', exist_ok=True)
        
        models = list(results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_names):
            ax = axes[i]
            values = [results[model][metric] for model in models]
            
            bars = ax.bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax.set_title(f'{metric.upper()} 比较')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # 在柱子上添加数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_xticklabels(models, rotation=45, ha='right')
        
        # 绘制混淆矩阵热图（使用最佳模型）
        best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
        ax = axes[5]
        
        # 获取最佳模型的混淆矩阵
        cm = np.array([
            [results[best_model_name]['true_negative'], results[best_model_name]['false_positive']],
            [results[best_model_name]['false_negative'], results[best_model_name]['true_positive']]
        ])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['预测非违约', '预测违约'],
                    yticklabels=['实际非违约', '实际违约'])
        ax.set_title(f'{best_model_name} 混淆矩阵')
        
        # 保存图表
        plt.tight_layout()
        chart_path = 'report/figures/model_comparison.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"模型比较图已保存: {chart_path}")
        
    except Exception as e:
        logger.error(f"绘制模型比较图失败: {str(e)}")
        raise

def plot_roc_curves(models, X_test, y_test):
    """绘制ROC曲线"""
    logger.info("绘制ROC曲线...")
    
    try:
        # 确保目录存在
        os.makedirs('report/figures', exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        
        # 绘制每个模型的ROC曲线
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                # 确保数据是数值类型
                X_test_processed = ensure_numeric_data(X_test.copy())
                
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
                logger.debug(f"模型{name}: AUC = {auc_score:.3f}")
        
        plt.xlabel('假正例率 (False Positive Rate)')
        plt.ylabel('真正例率 (True Positive Rate)')
        plt.title('ROC曲线比较')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        chart_path = 'report/figures/roc_curves.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC曲线图已保存: {chart_path}")
        
    except Exception as e:
        logger.error(f"绘制ROC曲线失败: {str(e)}")
        raise

def plot_feature_importance(model, feature_names):
    """绘制特征重要性/系数图"""
    logger.info("分析特征重要性...")
    
    try:
        if hasattr(model, 'coef_'):  # 逻辑回归
            importance = model.coef_[0]
            title = '逻辑回归特征系数'
            color = 'steelblue'
            logger.debug("使用逻辑回归系数")
        elif hasattr(model, 'feature_importances_'):  # 树模型
            importance = model.feature_importances_
            title = '特征重要性'
            color = 'forestgreen'
            logger.debug("使用特征重要性")
        else:
            logger.warning("模型不支持特征重要性分析")
            return
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # 记录特征重要性统计
        importance_stats = importance_df['importance'].describe()
        logger.debug(f"特征重要性统计:\n{importance_stats}")
        
        # 取最重要的20个特征
        top_features = importance_df.tail(20)
        logger.info(f"最重要的5个特征:")
        for _, row in top_features.tail(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 确保目录存在
        os.makedirs('report/figures', exist_ok=True)
        
        # 绘制水平条形图
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'], color=color)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('重要性/系数')
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, v in enumerate(top_features['importance']):
            plt.text(v, i, f' {v:.4f}', va='center')
        
        # 保存图表
        plt.tight_layout()
        chart_path = 'report/figures/feature_importance.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"特征重要性图已保存: {chart_path}")
        
        # 打印业务洞见
        print_business_insights(importance_df)
        
    except Exception as e:
        logger.error(f"分析特征重要性失败: {str(e)}")
        raise

def print_business_insights(importance_df):
    """打印业务洞见"""
    logger.info("\n生成业务洞见...")
    
    try:
        # 获取最重要的5个正特征和负特征
        positive_features = importance_df[importance_df['importance'] > 0].tail(5)
        negative_features = importance_df[importance_df['importance'] < 0].head(5)
        
        logger.info("\n1. 增加违约风险的关键因素（正系数）:")
        for _, row in positive_features.iterrows():
            logger.info(f"   • {row['feature']}: 系数 = {row['importance']:.4f}")
            logger.info(f"     含义: 该特征值越高，违约风险越大")
        
        logger.info("\n2. 降低违约风险的关键因素（负系数）:")
        for _, row in negative_features.iterrows():
            logger.info(f"   • {row['feature']}: 系数 = {row['importance']:.4f}")
            logger.info(f"     含义: 该特征值越高，违约风险越小")
        
        logger.info("\n3. 风险管理建议:")
        logger.info("   • 重点关注DTI（债务收入比）高的申请人")
        logger.info("   • 利率较高的贷款需要更严格的审批")
        logger.info("   • 收入稳定的申请人风险较低")
        logger.info("   • 信用等级（grade）是最重要的预测因素之一")
        logger.info("   • 建议建立综合评分卡，结合多个风险因素")
        
    except Exception as e:
        logger.error(f"生成业务洞见失败: {str(e)}")
        raise

def save_predictions(model, X_test, y_test, threshold=0.5):
    """
    保存预测结果
    owner: [F同学学号] + [F同学姓名]
    """
    logger.info("保存预测结果...")
    start_time = time.time()
    
    try:
        # ====== 新增：数据预处理 ======
        X_test_processed = ensure_numeric_data(X_test.copy())
        # ============================
        
        # 生成预测
        logger.debug(f"使用阈值: {threshold}")
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'true_label': y_test.values,
            'predicted_label': y_pred,
            'default_probability': y_pred_proba,
            'prediction_correct': (y_test.values == y_pred).astype(int)
        })
        
        # 重置索引以匹配原始数据
        results_df.index = X_test.index
        
        # 添加一些关键特征以便分析
        key_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'grade']
        for feature in key_features:
            if feature in X_test.columns:
                results_df[feature] = X_test[feature]
        
        # 计算预测置信度
        results_df['prediction_confidence'] = np.abs(y_pred_proba - 0.5) * 2
        
        # 保存到CSV
        results_df.to_csv(Config.OUTPUT_PREDICTIONS, index=True)
        
        save_time = time.time() - start_time
        
        logger.info(f"预测结果保存完成:")
        logger.info(f"  保存路径: {Config.OUTPUT_PREDICTIONS}")
        logger.info(f"  记录数: {len(results_df)}")
        logger.info(f"  阈值: {threshold}")
        logger.info(f"  保存耗时: {save_time:.2f}秒")
        
        # 分析预测结果
        analyze_predictions(results_df)
        
        # 分析不同阈值的影响
        analyze_thresholds(y_test, y_pred_proba)
        
        return results_df
        
    except Exception as e:
        logger.error(f"保存预测结果失败: {str(e)}")
        raise

def analyze_predictions(predictions_df):
    """分析预测结果"""
    logger.debug("分析预测结果...")
    
    try:
        # 基本统计
        total = len(predictions_df)
        correct = predictions_df['prediction_correct'].sum()
        accuracy = correct / total
        
        logger.info(f"预测结果分析:")
        logger.info(f"  总预测数: {total}")
        logger.info(f"  正确预测数: {correct}")
        logger.info(f"  预测准确率: {accuracy:.4f}")
        
        # 违约预测统计
        predicted_defaults = predictions_df['predicted_label'].sum()
        actual_defaults = predictions_df['true_label'].sum()
        
        logger.info(f"  预测违约数: {predicted_defaults}")
        logger.info(f"  实际违约数: {actual_defaults}")
        
        # 不同置信度的准确率
        confidence_bins = pd.cut(predictions_df['prediction_confidence'], 
                                 bins=[0, 0.3, 0.6, 0.8, 1.0])
        confidence_accuracy = predictions_df.groupby(confidence_bins)['prediction_correct'].mean()
        
        logger.info(f"不同置信度的准确率:")
        for bin_range, acc in confidence_accuracy.items():
            logger.info(f"  置信度{bin_range}: {acc:.3f}")
        
    except Exception as e:
        logger.warning(f"预测结果分析失败: {str(e)}")

def analyze_thresholds(y_true, y_pred_proba):
    """分析不同阈值的影响"""
    logger.info("进行阈值分析...")
    
    try:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        threshold_results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
        
        # 打印结果
        logger.info("\n阈值分析结果:")
        logger.info(f"{'阈值':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
        logger.info("-" * 70)
        for result in threshold_results:
            logger.info(f"{result['threshold']:<8.2f} {result['precision']:<8.3f} "
                       f"{result['recall']:<8.3f} {result['f1']:<8.3f} "
                       f"{result['tp']:<6} {result['fp']:<6} {result['tn']:<6} {result['fn']:<6}")
        
        logger.info("\n阈值选择建议:")
        logger.info("• 低阈值(0.3-0.4): 高召回率，适合不想漏掉任何违约的情况")
        logger.info("• 中阈值(0.5): 平衡精确率和召回率，一般用途")
        logger.info("• 高阈值(0.6-0.7): 高精确率，适合资源有限时只处理高风险客户")
        
        # 找到最佳阈值（按F1分数）
        best_by_f1 = max(threshold_results, key=lambda x: x['f1'])
        logger.info(f"推荐阈值（按F1分数）: {best_by_f1['threshold']:.2f} (F1 = {best_by_f1['f1']:.3f})")
        
        # 保存阈值分析结果
        threshold_df = pd.DataFrame(threshold_results)
        threshold_path = 'outputs/threshold_analysis.csv'
        threshold_df.to_csv(threshold_path, index=False)
        logger.info(f"阈值分析结果已保存: {threshold_path}")
        
    except Exception as e:
        logger.error(f"阈值分析失败: {str(e)}")
        raise

    
def ensure_numeric_data(X):
    """
    确保数据框的所有列都是数值类型
    Args:
        X: 输入数据框
    Returns:
        处理后的数值型数据框
    """
    X_processed = X.copy()
    
    logger.debug(f"原始数据类型统计:")
    for dtype, count in X_processed.dtypes.value_counts().items():
        logger.debug(f"  {dtype}: {count}列")
    
    # 处理日期时间列
    datetime_cols = X_processed.select_dtypes(include=['datetime', 'datetime64', 'timedelta']).columns
    if len(datetime_cols) > 0:
        logger.info(f"发现日期时间列: {list(datetime_cols)}")
        for col in datetime_cols:
            # 方法1: 转换为时间戳（秒数）
            try:
                X_processed[col] = pd.to_datetime(X_processed[col]).astype('int64') / 10**9
                logger.debug(f"  列 '{col}' 已转换为时间戳")
            except Exception as e:
                logger.warning(f"  列 '{col}' 日期转换失败: {e}")
                # 方法2: 如果转换失败，使用序号编码
                X_processed[col] = pd.factorize(X_processed[col])[0]
    
    # 处理布尔列
    bool_cols = X_processed.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        logger.info(f"发现布尔列: {list(bool_cols)}")
        for col in bool_cols:
            X_processed[col] = X_processed[col].astype(int)
            logger.debug(f"  列 '{col}' 已转换为整数")
    
    # 处理对象/字符串列
    object_cols = X_processed.select_dtypes(include=['object', 'category']).columns
    if len(object_cols) > 0:
        logger.info(f"发现对象类型列: {list(object_cols)}")
        for col in object_cols:
            try:
                # 尝试转换为数值
                X_processed[col] = pd.to_numeric(X_processed[col], errors='raise')
                logger.debug(f"  列 '{col}' 已转换为数值")
            except:
                # 如果是分类变量，使用标签编码
                X_processed[col] = pd.factorize(X_processed[col])[0]
                logger.debug(f"  列 '{col}' 已使用标签编码转换为数值")
    
    # 最后确保所有列都是数值类型
    X_processed = X_processed.apply(pd.to_numeric, errors='coerce')
    
    # 检查最终数据类型
    final_dtypes = X_processed.dtypes.unique()
    logger.debug(f"处理后数据类型: {final_dtypes}")
    
    return X_processed