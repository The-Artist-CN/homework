Lending Club 借贷违约风险评估项目
📋 项目概述
项目背景
Lending Club 是全球最大的P2P借贷平台之一，为个人和企业提供在线借贷服务。本项目基于Lending Club历史贷款数据，构建一个放款前可用的违约预测模型，用于评估个人信用贷款的违约风险。

应用场景与价值：

风险定价：根据违约概率确定贷款利率

授信决策：辅助批准/拒绝贷款申请

风险控制：识别高风险客户，降低坏账率

额度设定：基于风险评分设定贷款额度

项目目标
构建一个能够在放款前预测借款人违约概率的机器学习模型

识别影响违约的关键因素，提供业务洞见

为信贷决策提供数据支持，降低金融风险

👥 团队分工
角色	成员	学号	主要职责	完成状态
A. 数据读取与清洗	[A同学姓名]	[A同学学号]	clean_data	✅
B. 数据探索	[B同学姓名]	[B同学学号]	explore_data	✅
C. 特征工程	[C同学姓名]	[C同学学号]	feature_engineering	✅
D. 数据切分	[D同学姓名]	[D同学学号]	split_data	✅
E. 模型训练	[E同学姓名]	[E同学学号]	train_model	✅
F. 评估与导出	[F同学姓名]	[F同学学号]	evaluate_model / save_predictions	✅
G. 可视化与报告	[G同学姓名]	[G同学学号]	图表制作、报告编写	✅
H. 项目答辩	[H同学姓名]	[H同学学号]	PPT制作、项目汇报	✅
I. 数据探索辅助	[I同学姓名]	[I同学学号]	协助数据理解、可视化探索	✅
J. 特征工程辅助	[J同学姓名]	[J同学学号]	协助特征构造、验证	✅
代码规范：

每位成员在自己负责的函数顶部标注 (owner 学号)+姓名

不得修改他人代码，合并由队长负责

完成TODO后改为DONE状态


🏗️ 项目结构
text
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
│   ├── lc.csv                     # 样本数据文件
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
|
│
└── ai_usage_log.md                    # AI使用记录


🚀 快速开始
环境要求
Python 3.8+

4GB以上内存

2GB以上磁盘空间

安装步骤
克隆项目

bash
git clone <repository-url>
cd lending_club_project


# 方法2：手动安装依赖
pip install -r requirements.txt

# 创建必要目录
mkdir -p data outputs/models report/figures logs
准备数据文件

bash
# 将以下文件放入 data/ 目录：
# 1. lc.csv (Lending Club数据子集)
# 2. LCDataDictionary.xlsx (数据字典)
# 如果没有真实数据，程序会自动生成示例数据
运行命令
运行完整项目流程

bash
python main.py
运行测试套件


bash
# 查看预测结果
cat outputs/predictions.csv | head -10

# 查看模型性能比较
cat outputs/model_comparison.csv

# 查看日志文件
ls -la logs/
tail -f logs/lending_club_*.log
项目执行流程
text
1. 数据加载 → 2. 数据清洗 → 3. 数据探索 → 
4. 特征工程 → 5. 数据分割 → 6. 模型训练 → 
7. 模型评估 → 8. 结果导出
🔧 技术细节
数据处理流程
数据筛选：2013-2014年、36个月期限

目标变量映射：二分类标签转换

特征筛选：剔除贷后信息字段

缺失值处理：数值列中位数填充，类别列众数填充

异常值处理：IQR方法检测和截断

特征工程：创建至少2个新特征（对数变换、分箱等）

编码处理：One-Hot编码、标签编码、频率编码

模型训练
模型类型	用途	参数设置
逻辑回归	基线模型	C=1.0, class_weight='balanced'
随机森林	对比模型	n_estimators=100, max_depth=10
决策树	对比模型	max_depth=8, class_weight='balanced'
评估指标
指标	公式	业务意义
准确率	(TP+TN)/(TP+TN+FP+FN)	总体预测正确率
精确率	TP/(TP+FP)	预测违约中的真正违约比例
召回率	TP/(TP+FN)	实际违约中被正确识别的比例
F1分数	2×精确率×召回率/(精确率+召回率)	精确率和召回率的调和平均
ROC-AUC	ROC曲线下面积	模型整体区分能力
阈值选择
阈值	精确率	召回率	适用场景
0.3-0.4	较低	较高	不想漏掉任何违约
0.5	平衡	平衡	一般用途
0.6-0.7	较高	较低	资源有限，只处理高风险
📈 预期输出
文件输出
预测结果：outputs/predictions.csv

true_label: 实际标签

predicted_label: 预测标签

default_probability: 违约概率

prediction_confidence: 预测置信度

模型文件：outputs/models/

logistic_regression.pkl

random_forest.pkl

decision_tree.pkl

分析报告：report/figures/

6-8张关键可视化图表

模型性能比较图

特征重要性分析

控制台输出
text
============================================================
Lending Club借贷违约风险评估项目
============================================================

[步骤1] 数据加载...
原始数据形状: (XXXX, XX)
✓ 关键列检查通过

[步骤2] 数据清洗...
筛选后记录数: XXXX
目标变量分布: 违约19.8%，非违约80.2%
✓ 数据清洗完成

[步骤8] 结果导出...
预测结果已保存至: outputs/predictions.csv
✓ 项目执行完成！
📋 交付物清单
必交材料
代码仓库：完整的可复现代码

项目报告：6-8页PDF报告（问题描述、方法、结果、分析）

答辩幻灯：6-8页PPT演示文稿

运行结果：可复现的预测输出

技术要点
✅ 完整的端到端机器学习流程

✅ 至少2个模型对比（逻辑回归+随机森林/决策树）

✅ 详细的日志记录和错误处理

✅ 可复现的结果（固定随机种子）

✅ 业务洞见和阈值分析

✅ 完整的文档说明

🔍 故障排除
常见问题
导入错误：ModuleNotFoundError

bash
# 确保在项目根目录运行
cd lending_club_project

# 安装所有依赖
pip install -r requirements.txt

# 运行修复脚本
python fix_project.py
数据文件不存在

bash
# 检查数据文件
ls -la data/

# 如果没有数据，程序会自动生成示例数据
# 或者手动创建示例数据：
python -c "
import pandas as pd
import numpy as np
n=1000
df=pd.DataFrame({
    'loan_status':np.random.choice(['Fully Paid','Charged Off'],n,p=[0.8,0.2]),
    'loan_amnt':np.random.randint(5000,35000,n),
    'int_rate':np.random.uniform(5,30,n),
    'annual_inc':np.random.randint(30000,150000,n)
})
df.to_csv('data/sample.csv',index=False)
print('示例数据已创建')
"
内存不足

bash
# 减少数据量
# 修改 config/config.py 中的配置
# 或使用较小的样本数据
依赖包版本问题

bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 重新安装依赖
pip install --upgrade pip
pip install -r requirements.txt
获取帮助
查看详细日志：logs/ 目录下的日志文件

运行诊断脚本：python diagnose.py

检查配置文件：config/config.py

查看错误信息：程序运行时的完整Traceback

📚 参考资料
数据字典
完整字段说明：data/LCDataDictionary.xlsx

Lending Club官方数据说明：https://www.lendingclub.com/info/download-data.action

技术文档
Scikit-learn文档：https://scikit-learn.org

Pandas文档：https://pandas.pydata.org

Matplotlib文档：https://matplotlib.org

相关研究
信用评分模型研究

机器学习在金融风控中的应用

不平衡数据分类方法

📄 许可证
本项目仅供教学使用，数据来源于Lending Club公开数据。请遵守数据使用协议和学术诚信原则。

🏆 项目亮点
业务导向：紧密结合实际信贷业务场景

完整流程：从数据清洗到模型部署的全流程实现

可解释性：提供特征重要性和业务洞见

可复现性：固定随机种子，确保结果一致

团队协作：清晰的模块划分和分工合作

文档完整：详细的说明文档和错误处理

最后更新：2024年1月
项目状态：开发完成 ✅
维护团队：[团队名称]
联系方式：[团队邮箱或联系方式]

祝您使用愉快！如有问题，请查看日志文件或联系项目团队。