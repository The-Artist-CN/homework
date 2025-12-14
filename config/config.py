# config/config.py
"""
配置参数文件
owner: [队长学号] + [队长姓名]
"""
import random
import numpy as np
import logging
import os
import sys

# 添加 config 目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_logging import get_logger

# 获取logger
logger = get_logger('config')

class Config:
    """项目配置类"""
    
    # ========== 基本设置 ==========
    # 随机种子
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # 项目名称
    PROJECT_NAME = "Lending Club 借贷违约风险评估"
    VERSION = "1.0.0"
    
    # ========== 调试设置 ==========
    # 调试模式
    DEBUG_MODE = True  # 设置为True开启DEBUG模式
    
    # 详细日志选项
    LOG_DETAILED_METRICS = True  # 记录详细指标
    LOG_DATA_SAMPLES = True      # 记录数据样本
    LOG_FEATURE_DETAILS = True   # 记录特征详情
    LOG_MODEL_PARAMS = True      # 记录模型参数
    LOG_EXECUTION_TIME = True    # 记录执行时间 - 新增
    
    # ========== 数据设置 ==========
    # 关键列名
    KEY_COLUMNS = [
        'loan_status',      # 目标变量
        'issue_d',          # 发放日期
        'term',             # 贷款期限
        'grade',            # 信用等级
        'loan_amnt',        # 贷款金额
        'int_rate',         # 利率
        'annual_inc',       # 年收入
        'dti',              # 债务收入比
        'revol_util',       # 循环利用率
        'emp_length',       # 雇佣时长
        'home_ownership',   # 房产状况
        'purpose',          # 贷款用途
        'addr_state'        # 所在州
    ]
    
    # ========== 目标变量映射 ==========
    TARGET_MAPPING = {
        'Charged Off': 1,                                           # 违约
        'Default': 1,                                               # 违约
        'Does not meet the credit policy. Status:Charged Off': 1,   # 违约
        'Fully Paid': 0,                                            # 非违约
        'Does not meet the credit policy. Status:Fully Paid': 0,    # 非违约
        'Current': None,                                            # 排除
        'In Grace Period': None,                                    # 排除
        'Late (31-120 days)': None,                                 # 排除
        'Late (16-30 days)': None                                   # 排除
    }
    
    # ========== 样本选择 ==========
    # 时间筛选
    ISSUE_YEAR_START = 2013
    ISSUE_YEAR_END = 2014
    TERM = "36 months"
    
    # ========== 可用字段原则 ==========
    # 需要剔除的贷后字段关键词
    POST_LOAN_KEYWORDS = [
        'recover', 'settlement', 'pymnt', 'total_rec', 'out_prncp',
        'last_pymnt', 'next_pymnt', 'collection', 'debt_settlement',
        'hardship', 'payment_plan', 'disbursement', 'hardship_',
        'settlement_', 'deferral', 'orig_projected'
    ]
    
    # 需要保留的放款前字段
    PRE_LOAN_COLUMNS = [
        # 贷款基本信息
        'id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
        'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
        'verification_status', 'issue_d', 'loan_status', 'purpose',
        'title', 'zip_code', 'addr_state', 'dti',
        
        # 信用历史
        'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high',
        'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
        'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'initial_list_status',
        
        # 其他信用信息
        'collections_12_mths_ex_med', 'mths_since_last_major_derog',
        'policy_code', 'application_type', 'acc_now_delinq',
        'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il',
        'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
        'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
        'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
        'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
        'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
        'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
        'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
        'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
        'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
        'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
        'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
        'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
        'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies',
        'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
        'total_il_high_credit_limit'
    ]
    
    # ========== 数据分割设置 ==========
    TEST_SIZE = 0.3
    USE_STRATIFIED_SPLIT = True  # 使用分层采样
    
    # ========== 模型设置 ==========
    MODELS = {
        'logistic_regression': {
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': RANDOM_SEED,
            'solver': 'liblinear'
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        },
        'decision_tree': {
            'max_depth': 8,
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED
        }
    }
    
    # 要训练的模型
    TRAIN_MODELS = ['logistic_regression', 'random_forest', 'decision_tree']
    
    # ========== 特征工程设置 ==========
    # 新增特征配置
    FEATURE_ENGINEERING = {
        'log_annual_inc': True,  # 年收入对数变换
        'dti_bins': True,        # DTI分箱
        'revol_util_groups': True,  # 循环利用率分组
        'loan_to_income': True,  # 贷款收入比
        'grade_numeric': True,   # 信用等级数值化
        'emp_length_numeric': True  # 雇佣时长数值化
    }
    
    # DTI分箱设置
    DTI_BINS = [-np.inf, 10, 20, 30, np.inf]
    DTI_LABELS = ['low', 'medium', 'high', 'very_high']
    
    # 循环利用率分组
    REVOL_UTIL_BINS = [-np.inf, 30, 70, 90, np.inf]
    REVOL_UTIL_LABELS = ['low', 'medium', 'high', 'very_high']
    
    # 缺失值处理
    MISSING_THRESHOLD = 0.5  # 缺失率超过50%的列删除
    NUMERIC_FILL_STRATEGY = 'median'  # 数值列用中位数填充
    CATEGORICAL_FILL_STRATEGY = 'mode'  # 类别列用众数填充
    
    # 异常值处理
    OUTLIER_IQR_MULTIPLIER = 1.5  # IQR倍数
    
    # 特征缩放
    SCALE_FEATURES = True
    SCALER_TYPE = 'standard'  # 标准化
    
    # ========== 评估设置 ==========
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    DEFAULT_THRESHOLD = 0.5
    THRESHOLDS_TO_ANALYZE = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # ========== 路径设置 ==========
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    DATA_PATH = os.path.join(DATA_DIR, "lc.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    OUTPUT_MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    OUTPUT_PREDICTIONS = os.path.join(OUTPUT_DIR, "predictions.csv")
    REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
    REPORT_FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
    
    # ========== 日志设置 ==========
    LOG_LEVEL = 'DEBUG' if DEBUG_MODE else 'INFO'
    LOG_TO_FILE = True
    LOG_TO_CONSOLE = True
    LOG_MAX_FILE_SIZE = 10  # MB
    LOG_BACKUP_COUNT = 5
    LOG_SEPARATE_DEBUG_FILE = True  # 单独的调试日志文件
    
    @classmethod
    def init(cls):
        """初始化配置"""
        logger.info("=" * 60)
        logger.info(f"{cls.PROJECT_NAME} - 版本 {cls.VERSION}")
        logger.info("=" * 60)
        
        if cls.DEBUG_MODE:
            logger.info("🔧 调试模式已启用")
        
        logger.info("配置初始化:")
        logger.info(f"  随机种子: {cls.RANDOM_SEED}")
        logger.info(f"  数据筛选: {cls.ISSUE_YEAR_START}-{cls.ISSUE_YEAR_END}, {cls.TERM}")
        logger.info(f"  测试集比例: {cls.TEST_SIZE}")
        logger.info(f"  训练模型: {', '.join(cls.TRAIN_MODELS)}")
        
        # 创建必要目录
        cls.create_directories()
        
        # 验证配置
        cls.validate()
        
        logger.info("✓ 配置初始化完成")
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.OUTPUT_MODELS_DIR,
            cls.REPORT_DIR,
            cls.REPORT_FIGURES_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"创建/确认目录: {directory}")
        
        logger.info("目录结构创建完成")
    
    @classmethod
    def validate(cls):
        """验证配置参数"""
        logger.info("验证配置参数...")
        
        try:
            # 基本验证
            assert cls.TEST_SIZE > 0 and cls.TEST_SIZE < 1, "TEST_SIZE必须在0和1之间"
            assert cls.ISSUE_YEAR_START <= cls.ISSUE_YEAR_END, "起始年份必须小于等于结束年份"
            
            # 路径验证
            assert os.path.isdir(cls.PROJECT_ROOT), f"项目根目录不存在: {cls.PROJECT_ROOT}"
            
            # 模型验证
            assert len(cls.TRAIN_MODELS) > 0, "至少需要训练一个模型"
            
            logger.info("✓ 配置验证通过")
            return True
            
        except AssertionError as e:
            logger.error(f"配置验证失败: {e}")
            raise
    
    @classmethod
    def get_pre_loan_columns(cls):
        """获取放款前可用字段列表"""
        return cls.PRE_LOAN_COLUMNS

# 初始化配置
Config.init()