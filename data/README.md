📊 数据说明
数据来源
原始数据：Lending Club 2007年6月-2018年12月公开数据

数据规模：约200万条记录，145个字段

本项目使用：2013-2014年发放的36个月期贷款子集

数据筛选规则
筛选条件	说明
时间范围	2013年1月1日 - 2014年12月31日
贷款期限	36个月（已结束，可观察最终结果）
观察时点	2018年底前已结清或违约
目标变量定义
违约标签映射表：

loan_status (原始)	二分类标签	说明
Charged Off	1 (违约)	已核销的坏账
Default	1 (违约)	违约
Does not meet the credit policy. Status:Charged Off	1 (违约)	不符合信贷政策的坏账
Fully Paid	0 (非违约)	已全额还款
Does not meet the credit policy. Status:Fully Paid	0 (非违约)	不符合信贷政策的全额还款
Current	排除	仍在还款中
In Grace Period	排除	宽限期
Late (31-120 days)	排除	逾期31-120天
Late (16-30 days)	排除	逾期16-30天
口径说明：

违约 = 1：借款人未能按时足额偿还贷款

非违约 = 0：借款人按时全额偿还贷款

排除样本：贷款尚未结束或处于特殊状态

可用字段原则
重要原则：只使用放款前可获得的信息

剔除的贷后字段类型：

还款相关：recover*, settlement*, pymnt*, last_pymnt*, next_pymnt*

未偿还本金：out_prncp*, total_rec*

催收相关：collection*, debt_settlement*

困难援助：hardship*

还款计划：payment_plan*

资金拨付：disbursement*

保留的放款前信息：

借款人基本信息：收入、职业、信用记录

贷款申请信息：金额、利率、期限、用途

信用评估信息：信用等级、债务收入比

历史记录：逾期记录、查询记录

关键字段释义
字段名	类型	说明	业务意义
loan_amnt	数值	贷款金额	贷款规模
term	类别	贷款期限	还款周期
int_rate	数值	利率	资金成本
grade	类别	信用等级	风险评级
sub_grade	类别	信用子等级	细化评级
emp_length	类别	雇佣时长	收入稳定性
home_ownership	类别	房产状况	资产状况
annual_inc	数值	年收入	还款能力
verification_status	类别	收入验证状态	信息可信度
purpose	类别	贷款用途	资金去向
dti	数值	债务收入比	负债压力
delinq_2yrs	数值	过去2年逾期次数	信用历史
fico_range_low	数值	FICO信用分下限	信用评分
open_acc	数值	当前账户数	信贷活跃度
revol_bal	数值	循环余额	信用卡使用
revol_util	数值	循环利用率	信用卡使用率
total_acc	数值	总账户数	信贷经验
initial_list_status	类别	初始挂牌状态	发行状态