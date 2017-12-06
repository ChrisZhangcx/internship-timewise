# -*- coding:utf8 -*-
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def generate_select_order(attributes, table, condition=None):
    return "select "+attributes+" from "+table+" "+condition


"""
数据库读取命令的配置
"""
#title = ['con_date', 'c1', 'c3', 'c4', 'c5', 'c6', 'c7', 'c12', 'cb', 'cpb']
title = ['con_date', 'c13', 'c9']
#title = ['con_date', 'cgb', 'cgpb', 'cgg', 'cgpeg']
#order = generate_select_order("con_date, c1, c3, c4, c5, c6, c7, c12, cb, cpb", "con_forecast_idx", "where stock_code='000300' and rpt_date='2013'")
order = generate_select_order("con_date, c13, c9", "con_forecast_c2_idx", "where stock_code='000300'")
#order = generate_select_order("con_date, cgb, cgpb, cgg, cgpeg", "con_forecast_c3_idx", "where stock_code='000300'")



"""
模型参数配置
"""
validate_times = 100
accept_ratio = 0.15     # 对于accuracy约为0.1-0.15，对weight-xu约为0.03-0.05
delete_ratio = 0.5
test_size = 0.2
max_check_days = 15
kinds = 5
gnb = GaussianNB()
model = gnb
selection_method = 'accuracy'


"""
写入测试结果到文件参数配置
"""
# 设置因子信息：因子来源表+因子名称
#factor_source = "一致预期数据表con_forecast_idx".decode('utf8').encode('gbk')
factor_source = "一致预期衍生指标计算con_forecast_c2_idx".decode('utf8').encode('gbk')
#factor_source = "一致预期滚动衍生数据表con_forecast_c3_idx".decode('utf8').encode('gbk')
#factor_name = ['一致预期EPS_c1', '2年复合增长率_c3', '一致预期归属母公司净利润_c4', '一致预期PE_c5', '一致预期PE/G_c6',
#               '一致预期净利同比_c7', '一致预期ROE_c12', '一致预期净资产_cb', '一致预期PB_cpb', '干扰数据']
factor_name = ['滚动净利润_c13', '滚动PE_c9', '干扰数据']
#factor_name = ['滚动净资产_cgb', '滚动市净率（滚动PB）_cgpb', '滚动净利润复合增长率_cgg', '滚动PEG_cgpeg', '干扰数据']
