# -*- coding:utf8 -*-
# 获取并保存用来训练模型的数据、用来进行测试的数据
"""
1. 读取数据
2. 读取数据库，匹配日期，生成最终模型生成数据与最终因子集
3. 保存训练、测试集到pickle，保存最终因子集
"""
import pickle
from db import connect, disconnect
import numpy as np
import csv
from config import title, order


dname = ['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额', '涨跌幅', '振幅', '均价', '总市值',      # 0-10
         '市盈率', '市净率', '市销率', '市现率', '股息率', 'ADTM', 'ATR', 'BBI', 'BBIBOLL', 'BIAS', 'BOLL',      #  11-21
         'CCI', 'CDP', 'DMA', 'DMI', 'DPO', 'ENV', 'EXPMA', 'PVT', 'SOBV', 'TAPI', 'VMA', 'VMACD', 'VSTD',   # 22-34
         'WVAD', '历史新低', '阶段新高', '历史新高', '阶段新低', '连涨天数', '连跌天数', '向上突破', '向下突破',      #  35-43
         '看涨看跌', '待预测']                                                                                 # 44-45
ddict = {}


def get_data():
    """
    读取存储的数据到ddict字典
    :return:
    """
    global ddict
    input_address = open('data_factor.txt', 'r')
    ddict = pickle.load(input_address)


# 连接数据库
cur = connect()


# 获取数据
get_data()

# 初始化训练与测试集
date_initial = np.array(ddict['日期'])
all_X = []
all_y = [float(x) for x in ddict['待预测']]
for i in range(1, 16):
    tx = np.array([float(x) for x in ddict[dname[i]]])
    all_X.append(tx)
all_X = np.transpose(all_X)
all_y = np.transpose([np.array(all_y)])

# 初始化因子集
cur.execute(order)
data = cur.fetchall()
data = np.array(data)
date_relative = data[:, 0]
print len(date_relative)

# 筛选存在相同数据的训练集、测试集、因子集
x = []
y = []
factors = []
order_date = []
for i in range(0, len(date_relative)):
    date = date_relative[i]
    index = np.where(date_initial == date)[0]
    if len(index) != 1:
        continue
    x.append(all_X[i])
    y.append(all_y[i])
    factors.append(data[i, 1:])
    order_date.append(date)
x = np.array(x)
y = np.array(y)
factors = np.array(factors)
order_date = np.transpose([order_date])

# 保存数据以便读取
input_file = open('mysql_factors', 'wb')
pickle.dump([x, y, factors], input_file)

# 保存测试因子数据到csv文件
csvfile = open('final_factors.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerow(title)
writer.writerows(np.hstack((order_date, factors)))
csvfile.close()

# 断开数据库连接
disconnect()
