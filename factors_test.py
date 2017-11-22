# -*- coding:utf8 -*-
# 测试因子
"""
1. 获取待测试的因子（数据库/本地保存的文件）
2. 加入扰动因子，以比较模型抗干扰性
3. 调整部分参数（因子有效性接受阈值）
4. 因子测试，使用函数输出测试结果
5*.保存测试结果到本地
"""
import pickle
import numpy as np
import csv
import datetime
import random


# 格式化输出模型因子测试的结果
def print_factor_test_result(result):
    print "因子测试结束"
    if result['method'] is None:
        print "当前因子有效性不显著\n"
    else:
        print "当前因子有效性显著"
        print "前查天数", result['days']
        print "处理方式", result['method']
        accuracy_list = result['accuracy_list']
        print "预测精度均值", accuracy_list.mean(), "预测精度标准差", accuracy_list.std()
        print ""


# 读取数据到变量
input = open('filtered_factors.txt', 'rb')
data = pickle.load(input)
input.close()
fs = data[0]
print fs.score, fs.std

factors = data[1]
# 在因子中加入扰动变量，便于比较模型的抗干扰性
turbe = np.transpose([range(0, factors.shape[0])])
random.shuffle(turbe)
factors = np.hstack((factors, turbe))
# 调整因子有效性的接受值
fs.change_ratio(accept_ratio=0.1)

# 生成因子测试报告
csvfile = open('final_test.csv', 'ab')
writer = csv.writer(csvfile)

# 写入测试时间
writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
# 写入模型基本信息
model_info = "模型准确度均值："+str(fs.score)+"，标准差："+str(fs.std)+"，有效阈值："+str(fs.score+fs.get_ratio()[0]*fs.std)
model_info = model_info.decode('utf8').encode('gbk')
writer.writerow([model_info])
# 设置因子信息：因子来源表+因子名称
#factor_source = "一致预期数据表con_forecast_idx".decode('utf8').encode('gbk')
factor_source = "一致预期衍生指标计算con_forecast_c2_idx".decode('utf8').encode('gbk')
#factor_source = "一致预期滚动衍生数据表con_forecast_c3_idx".decode('utf8').encode('gbk')
#factor_name = ['c1', 'c3', 'c4', 'c5', 'c6', 'c7', 'c12', 'cb', 'cpb', '干扰数据']
factor_name = ['滚动净利润_c13', '滚动PE_c9', '干扰数据']
# factor_name = ['cgb', 'cgpb', 'cgg', 'cgpeg', '干扰数据']

for i in range(0, len(factor_name)):
    factor_name[i] = factor_name[i].decode('utf8').encode('gbk')
# 写入因子信息
writer.writerow([factor_source])
factor_title = ['因子名称', '准确度均值', '前查天数', '处理方式', '是否接受']
for i in range(0, len(factor_title)):
    factor_title[i] = factor_title[i].decode('utf8').encode('gbk')
writer.writerow(factor_title)
for each_factor in range(0, factors.shape[1]):
    # 写入每个因子的基本信息
    t_factor = np.transpose([factors[:, each_factor]])
    record_result = fs.test_factor(fs.X_chosen, fs.y_chosen, factor=t_factor, update=False)
    if record_result['accuracy'] > fs.score+fs.get_ratio()[0]*fs.std:
        writer.writerow([factor_name[each_factor], record_result['accuracy'], record_result['days'],
                         record_result['method'], 1])
    else:
        writer.writerow([factor_name[each_factor], record_result['accuracy'], record_result['days'],
                         record_result['method'], 0])
csvfile.close()
