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
from config import factor_name, factor_source


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


# 读取测试因子到变量
input = open('mysql_factors', 'rb')
data = pickle.load(input)
input.close()
factors = data[2]
column = factors.shape[1]

# 打开因子测试报告写入文件
csvfile = open('final_test.csv', 'ab')
writer = csv.writer(csvfile)

# 写入测试时间
writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

for i in range(0, len(factor_name)):
    factor_name[i] = factor_name[i].decode('utf8').encode('gbk')
# 写入因子信息
writer.writerow([factor_source])
factor_title = ['因子名称', '准确度均值', '因子评分']
for i in range(0, len(factor_title)):
    factor_title[i] = factor_title[i].decode('utf8').encode('gbk')
for each_factor in range(0, column):
    print "当前判断第".decode('utf8').encode('gbk'), each_factor, "个因子".decode('utf8').encode('gbk')
    t_factor = np.transpose([factors[:, each_factor]])
    std_list_before = []
    accu_list_before = []
    accu_list_after = []
    for i in range(0, 15):
        print "判断第".decode('utf8').encode('gbk'), i, "个模型中因子的测试效果".decode('utf8').encode('gbk')
        factors = data[2]
        model_title = "base_model"+str(i)+".pkl"
        input_file = open(model_title, 'rb')
        fs = pickle.load(input_file)
        # 在因子中加入扰动变量，便于比较模型的抗干扰性
        turbe = np.transpose([range(0, factors.shape[0])])
        random.shuffle(turbe)
        factors = np.hstack((factors, turbe))
        fs.change_ratio(accept_ratio=0.1)

        accu_list_before.append(fs.score)
        std_list_before.append(fs.std)
        record_result = fs.test_factor(fs.X_chosen, fs.y_chosen, factor=t_factor, update=False)
        accu_list_after.append(record_result['accuracy'])
    std_list_before = np.array(std_list_before)
    accu_list_before = np.array(accu_list_before)
    accu_list_after = np.array(accu_list_after)
    print "原标准差均值".decode('utf8').encode('gbk'), std_list_before.mean()
    print "原精度均值".decode('utf8').encode('gbk'), accu_list_before.mean()
    print "后精度均值".decode('utf8').encode('gbk'), accu_list_after.mean()
    print "因子最终评分".decode('utf8').encode('gbk'), (accu_list_after.mean()-accu_list_before)/std_list_before.mean()

    # 写入模型基本信息
    model_info = "模型准确度均值："+str(accu_list_before.mean*())+"，标准差："+str(std_list_before.mean())
    model_info = model_info.decode('utf8').encode('gbk')
    writer.writerow([model_info])
    # 写入因子测试结果
    writer.writerow(factor_title)
    writer.writerow([factor_name[each_factor], accu_list_after.mean(),
                     (accu_list_after.mean()-accu_list_before)/std_list_before.mean()])
csvfile.close()
