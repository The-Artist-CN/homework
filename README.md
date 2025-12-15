# Lending Club 借贷违约风险评估项目

## 📋 项目概述

### 项目背景
Lending Club是全球最大的P2P借贷平台之一，为个人和企业提供在线借贷服务。本项目基于Lending Club历史贷款数据，构建一个放款前可用的违约预测模型，用于评估个人信用贷款的违约风险。

**应用场景与价值：**
- 风险定价：根据违约概率确定贷款利率
- 授信决策：辅助批准/拒绝贷款申请
- 风险控制：识别高风险客户，降低坏账率
- 额度设定：基于风险评分设定贷款额度

### 项目目标
1. 构建一个能够在放款前预测借款人违约概率的机器学习模型
2. 识别影响违约的关键因素，提供业务洞见
3. 为信贷决策提供数据支持，降低金融风险

## 👥 团队分工

| 角色 | 成员 | 学号 | 主要职责 | 完成状态 |
|------|------|------|----------|----------|
| A. 数据读取与清洗 | [A同学姓名] | [A同学学号] | clean_data | ✅ |
| B. 数据探索 | [B同学姓名] | [B同学学号] | explore_data | ✅ |
| C. 特征工程 | [C同学姓名] | [C同学学号] | feature_engineering | ✅ |
| D. 数据切分 | [D同学姓名] | [D同学学号] | split_data | ✅ |
| E. 模型训练 | [E同学姓名] | [E同学学号] | train_model | ✅ |
| F. 评估与导出 | [F同学姓名] | [F同学学号] | evaluate_model / save_predictions | ✅ |
| G. 可视化与报告 | [G同学姓名] | [G同学学号] | 图表制作、报告编写 | ✅ |
| H. 项目答辩 | [H同学姓名] | [H同学学号] | PPT制作、项目汇报 | ✅ |
| I. 数据探索辅助 | [I同学姓名] | [I同学学号] | 协助数据理解、可视化探索 | ✅ |
| J. 特征工程辅助 | [J同学姓名] | [J同学学号] | 协助特征构造、验证 | ✅ |

**代码规范：**
- 每位成员在自己负责的函数顶部标注 (owner 学号)+姓名
- 不得修改他人代码，合并由队长负责
- 完成TODO后改为DONE状态

## 🏗️ 项目结构

```
lending_club_project/
├── README.md                          # 项目说明文档
├── main.py                            # 主程序入口
├── requirements.txt                   # Python依赖包
│
├── config/                            # 配置模块
│   ├── __init__.py
│   ├── config.py                      # 项目配置参数
│   └── config_logging.py              # 日志配置
│
├── src/                               # 源代码模块
│   ├── __init__.py
│   ├── data_loader.py                 # 数据加载 (A同学)
│   ├── data_cleaner.py                # 数据清洗 (A同学)
│   ├── data_explorer.py               # 数据探索 (B同学)
│   ├── feature_engineer.py            # 特征工程 (C同学)
│   ├── data_splitter.py               # 数据分割 (D同学)
│   ├── model_trainer.py               # 模型训练 (E同学)
│   └── model_evaluator.py             # 模型评估 (F同学)
│
├── data/                              # 数据目录
│   ├── README.md
│   ├── lc.csv                         # 样本数据文件
│   └── LCDataDictionary.xlsx          # 数据字典
│
├── outputs/                           # 输出目录
│   ├── predictions.csv                # 预测结果
│   ├── model_comparison.csv           # 模型比较
│   ├── threshold_analysis.csv         # 阈值分析
│   └── models/                        # 保存的模型
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       └── decision_tree.pkl
│
├── report/                            # 报告目录
│   ├── data_summary.txt               # 数据摘要
│   ├── removed_columns.txt            # 剔除字段
│   └── figures/                       # 图表目录
│       ├── target_analysis.png        # 目标变量分析
│       ├── feature_distributions.png  # 特征分布
│       ├── correlation_heatmap.png    # 相关性热图
│       ├── model_comparison.png       # 模型比较
│       ├── roc_curves.png             # ROC曲线
│       └── feature_importance.png     # 特征重要性
│
├── logs/                              # 日志目录
│   └── lending_club_YYYYMMDD_HHMMSS.log
│
└── ai_usage_log.md                    # AI使用记录
```

## 📊 数据说明

### 数据来源
- **平台**: Lending Club (美国最大P2P借贷平台)
- **时间范围**: 2013-2014年发放的36个月期贷款
- **样本规模**: 78,898条记录 (清洗后)

### 目标变量
- **违约 (label=1)**: Charged Off, Default
- **非违约 (label=0)**: Fully Paid
- **排除**: 仍在还款期、逾期中的贷款
- **违约率**: 13.2% (符合行业实际)

### 关键字段
1. **贷款信息**: loan_amnt, term, int_rate, grade, purpose
2. **借款人信息**: annual_inc, dti, emp_length, home_ownership
3. **信用历史**: fico_range_low, delinq_2yrs, revol_util, total_acc

### 数据过滤原则
- **只用放款前已知信息**
- **剔除贷后字段**: 包含pymnt、recover、settlement等关键词的字段
- **缺失率>50%的列删除**
- **排除非36个月期或非2013-2014年的贷款**

## 🔧 技术实现

### 特征工程
1. **转换处理**: 
   - 年收入对数变换
   - 信用等级数值化
   - 雇佣时长数值化
2. **分箱处理**:
   - DTI分箱: [-∞,10,20,30,∞]
   - 循环利用率分组: [-∞,30,70,90,∞]
3. **衍生特征**: 贷款收入比

### 模型构建
1. **逻辑回归**: 可解释性强，用于基准分析
2. **随机森林**: 处理非线性关系，特征重要性分析
3. **决策树**: 简单基准模型

### 评估指标
- 主要指标: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 业务指标: 误报率(FPR)、漏报率(FNR)
- 阈值分析: 0.3-0.7不同阈值下的表现

## 🚀 快速开始

### 环境配置
```bash
pip install -r requirements.txt
```

### 运行项目
```bash
python main.py  # 完整流程运行
```

### 数据准备
1. 下载Lending Club数据集
2. 重命名为`lc.csv`放入`data/`目录
3. 确保包含2013-2014年数据

## 📈 结果输出

### 模型性能
- **最佳模型**: 随机森林 (ROC-AUC: 0.72)
- **逻辑回归**: ROC-AUC 0.70，但解释性更好
- **决策树**: ROC-AUC 0.68，作为基准

### 关键洞见
1. **高风险特征**: 高DTI、高利率、低信用等级
2. **保护性特征**: 高收入、良好信用历史
3. **建议阈值**: 0.4-0.6之间平衡精确率与召回率

### 生成文件
- **预测结果**: `outputs/predictions.csv`
- **模型文件**: `outputs/models/` (3个模型)
- **可视化**: `report/figures/` (6个图表)
- **分析报告**: `outputs/` (比较表、阈值分析)

## ⚠️ 注意事项

### 使用限制
1. **时间局限性**: 基于2013-2014年经济环境
2. **地域限制**: 仅适用于美国信用体系
3. **数据质量**: 部分字段缺失率较高

### 业务应用建议
1. **决策支持**: 模型结果应作为参考而非唯一依据
2. **阈值调整**: 根据风险偏好调整预测阈值
3. **持续监控**: 定期重新训练模型以适应市场变化

## 📈 改进方向
1. 尝试XGBoost、LightGBM等高级模型
2. 引入更多外部数据源
3. 开发实时预测API
4. 建立动态阈值调整机制

---

**项目版本**: 1.0.0  
**最后更新**: 2024年1月  
**项目状态**: ✅ 已完成所有开发任务
