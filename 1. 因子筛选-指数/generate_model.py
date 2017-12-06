# -*- coding:utf8 -*-
# 设置数据，调用FactorScreen实例进行测试
"""
1. 读取本地的训练、测试集数据。数据格式：x-m个样本*n个特征，y-m个样本*1个输出
2. 对y进行离散化
3. 选择模型，设置因子筛选器参数
4. 创建实例，进行模型生成
5. 保存数据：因子筛选器实例；模型的最终xy数据
"""
from FactorScreen import FactorFilter
from Preprocessing import ProcessingMethod
import pickle
from config import validate_times, accept_ratio, delete_ratio, test_size, max_check_days, kinds, model, selection_method
import random
from sklearn.model_selection import train_test_split


# 读取数据
output_file = open('mysql_factors', 'rb')
[x, y, factors] = pickle.load(output_file)
# 对y作离散化处理
y = ProcessingMethod.discretization(y, kinds=kinds)

# 配置因子筛选器实例
fs = FactorFilter(x, y, model=model, validate_times=validate_times, ar=accept_ratio, dr=delete_ratio,
                  test_size=test_size, max_check_days=max_check_days, method=selection_method, kinds=kinds)
for i in range(0, x.shape[1]):
    print "\n以第".decode('utf8').encode('gbk'), i, "个下标作为初始因子开始训练".decode('utf8').encode('gbk')
    fs.filter_factors(start_index=i)
    model_title = "base_model" + str(i) + ".pkl"
    output_file = open(model_title, 'wb')
    pickle.dump(fs, output_file)
    output_file.close()

# 保存实例，便于读取
data = (fs, factors)
output = open('filtered_factors.txt', 'wb')
pickle.dump(data, file=output)
output.close()


# 保存X_chosen与y_chosen到csv文件（已经是最终形式）
fs.save()

