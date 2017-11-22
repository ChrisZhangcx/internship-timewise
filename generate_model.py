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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from Preprocessing import ProcessingMethod
import pickle
import random
from sklearn.model_selection import train_test_split


# 格式化输出模型自训练的结果
def print_filter_result(fs):
    print "完成了模型的参数选择过程"
    print "最终输入集合", fs.X_chosen
    print "选择的参数下标依次为", fs.X_choose_step
    print "选择的参数前查天数为", fs.X_check_days
    print "选择的参数处理方式为", fs.X_choose_method
    print "最终输入集大小", fs.X_chosen.shape
    print "最终输出集大小", fs.y_chosen.shape


# 读取数据
output_file = open('mysql_factors', 'rb')
x, y, factors = pickle.load(output_file)

# 选择模型
gnb = GaussianNB()
dtc = DecisionTreeClassifier()

# 创建因子筛选器实例
validate_times = 100
accept_ratio = 0.15     # 对于accuracy约为0.1-0.15，对weight-xu约为0.03-0.05
delete_ratio = 0.5
test_size = 0.2
max_check_days = 15
kinds = 5
model = gnb

# 对y作离散化处理
y = ProcessingMethod.discretization(y, kinds=kinds)

fs = FactorFilter(x, y, model=model, validate_times=validate_times, ar=accept_ratio, dr=delete_ratio,
                  test_size=test_size, max_check_days=max_check_days, method='accuracy', kinds=kinds)
fs.filter_factors()
print_filter_result(fs)
print ""

# 模型最终效果测试
accuracy_final = fs.validate(fs.X_chosen, fs.y_chosen, bias=100)
print "交叉验证", validate_times, "次后的准确度均值为", accuracy_final.mean(), "准确度标准差为", accuracy_final.std()

# 保存实例，便于读取
data = (fs, factors)        # 要保存的数据
output = open('filtered_factors.txt', 'wb')
pickle.dump(data, file=output)
output.close()

# 保存X_chosen与y_chosen到csv文件（已经是最终形式）
fs.save()

