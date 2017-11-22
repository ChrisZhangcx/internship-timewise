# -*- coding:utf8 -*-
from Preprocessing import ProcessingMethod
from sklearn.model_selection import train_test_split
import numpy as np
import csv


class FactorFilter:
    __validate_times = 30       # 交叉验证次数（用于假设检验）
    __test_size = 0.2           # 交叉验证时的测试集比例
    __max_check_days = 10       # 最大向前查看天数
    __X = []                    # 输入集
    __y = []                    # 输出集
    __model = None              # 用来进行训练的模型
    __accept_ratio = 1.         # 加入因子时的标准差系数：即mean()>mean()+ratio*std()时加入因子
    __delete_ratio = 0.5        # 删除因子时的标准差系数：即mean()>mean()-ratio*std()时删除因子
    __kinds = 5                 # 如果输出集被离散化，其离散化的种类数
    __method = 'accuracy'       # 用来计算精确度的方法

    factors_filtered = 0        # 是否已进行因子筛选
    looking_forward_days = 0    # 当前需要删除样本的总数
    X_chosen = []               # 筛选后生成的输入矩阵
    y_chosen = []               # 筛选后生成的输出矩阵
    X_choose_step = []          # 因子被加入因子库的顺序（下标）
    X_choose_method = []        # 因子被加入时进行的数据预处理方式
    X_check_days = []           # 因子被加入时向前查看的天数
    score = 0.                  # 筛选后的分类准确度均值
    std = 0.                    # 筛选后的分类准确度标准差

    # 初始化函数
    def __init__(self, x, y, model, validate_times=30, max_check_days=10, test_size=0.2, ar=1., dr=0.5, kinds=5,
                 method='accuracy'):
        self.__X = x
        self.__y = y
        self.__model = model
        self.__validate_times = validate_times
        self.__test_size = test_size
        self.__max_check_days = max_check_days
        self.__accept_ratio = ar
        self.__delete_ratio = dr
        self.__kinds = kinds
        self.__method = method

    # 改变判断因子有效性的阈值
    def change_ratio(self, accept_ratio=None, delete_ratio=None):
        if accept_ratio is not None:
            self.__accept_ratio = accept_ratio
        if delete_ratio is not None:
            self.__delete_ratio = delete_ratio

    # 返回接受阈值与拒绝阈值
    def get_ratio(self):
        return self.__accept_ratio, self.__delete_ratio

    # 返回训练好的模型
    def get_model(self):
        return self.final_model

    # 对因子列表进行自动筛选，获得效果最好的因子组合
    def filter_factors(self):
        print "开始寻找最优因子组合，当前输入集大小为", self.__X.shape
        # 随机选择起始因子，并在原始数据中剔除该因子，初始化X_chosen与y_chosen矩阵
        start_factor_index = np.random.randint(0, self.__X.shape[1])
        self.X_chosen = np.transpose([ProcessingMethod.scale(self.__X[:, start_factor_index])])
        self.y_chosen = self.__y.ravel()
        self.X_choose_step.append(start_factor_index)
        self.X_check_days.append(0)
        self.X_choose_method.append(ProcessingMethod.scale.__name__)
        print "随机开始因子下标：", start_factor_index
        for col in range(0, self.__X.shape[1]):     # 遍历每个因子
            if col == start_factor_index:
                continue
            # 判断下一个因子的有效性
            print "\n开始判断下一个因子的有效性，当前判断下标为", col, "的因子"
            t_factor = np.transpose([self.__X[:, col]])
            print "传入X_chosen.shape=", self.X_chosen.shape, "y_chosen.shape=", self.y_chosen.shape, "因子.shape=", t_factor.shape
            result = self.test_factor(self.X_chosen, self.y_chosen, t_factor)
            if result['method'] is not None:        # 因子有效
                self.X_choose_step.append(col)
                self.X_choose_method.append(result['method'])
                self.X_check_days.append(result['days'])
        self.factors_filtered = 1
        # 将数据转换为ndarray类型，否则无法使用pickle存储
        self.X_choose_step = np.array(self.X_choose_step)
        self.X_check_days = np.array(self.X_check_days)
        self.X_choose_method = np.array(self.X_choose_method)
        # 保存训练后的模型，以进行模型可视化

    def test_factor(self, x, y, factor, update=True):
        """
        传入临时输入集、临时输出集和因子序列，判断因子（前查0-max_check_day天，标准化）是否有效，找到最有效的条件
        如果result字典中method=None，说明当前因子没有达到显著性差异的要求，无法被选择；
        :param x: 训练集
        :param y: 测试集
        :param factor: 因子序列(m*1)
        :param update: 标志位，因子有效时，是否将当前因子加入模型
        :return: 字典类型的最佳结果
        """
        # 计算原模型的准确度列表
        print "当前输入集的格式为", self.X_chosen.shape
        accu_before = self.validate(self.X_chosen, self.y_chosen)
        print "原模型均值与标准差", accu_before.mean(), accu_before.std()

        best_result = {'method':None, 'days':0, 'accuracy_list':None}       # 如果接受，保存最好结果
        record_result = {'method':None, 'days':0, 'accuracy':0.}            # 保存本次训练中的最好结果（无论是否接受）
        temp_best_accuracy = accu_before.mean()     # 如果接受的最好准确度
        # 切割因子集，使之与输入集大小相同
        factor = factor[self.looking_forward_days:, ]
        # 开始遍历寻找最优参数
        for days in range(0, self.__max_check_days+1):
            for method in [ProcessingMethod.scale, ProcessingMethod.centralization, ProcessingMethod.range_standard,
                           ProcessingMethod.discretization]:
                # 因子处理：前查+变换
                t_factor = ProcessingMethod.get_delta(factor=factor, days=days)
                for col in range(0, t_factor.shape[1]):
                    # 离散化处理方式单独进行判断
                    if method == ProcessingMethod.discretization:
                        t_factor[:, col] = np.transpose(method(np.transpose([t_factor[:, col]])))[0]
                    else:
                        t_factor[:, col] = method(t_factor[:, col])
                # 重新生成训练数据
                tx = np.hstack((x, t_factor))
                tx = tx[days:, ]
                ty = y[days:]
                # 进行交叉验证，生成结果序列
                accu_after = self.validate(tx, ty)
                tam = accu_after.mean()
                # 判断结果提升是否显著
                print "前查", days, "天", "最好均值", temp_best_accuracy, "当前均值", tam
                if tam > record_result['accuracy']:
                    record_result['method'] = method.__name__
                    record_result['days'] = days
                    record_result['accuracy'] = tam
                if accu_after.mean() > accu_before.mean()+self.__accept_ratio*accu_before.std():    # 通过了接受阈值
                    # 比当前最好结果更好（最好结果可能已经在该因子的之前循环中更新）
                    if accu_after.mean() > temp_best_accuracy:
                        print "结果提升显著，将", temp_best_accuracy, "更新到", accu_after.mean()
                        best_result['accuracy_list'] = accu_after
                        best_result['method'] = method.__name__
                        best_result['days'] = days
                        if update:
                            self.X_chosen = tx
                            self.y_chosen = ty
                            self.score = best_result['accuracy_list'].mean()
                            self.std = best_result['accuracy_list'].std()
                        temp_best_accuracy = accu_after.mean()
        # 如果使用有效的因子更新实例内的训练集与测试集，则更新前查天数与均值、标准差
        if update:
            self.looking_forward_days += best_result['days']
            return best_result
        else:
            return record_result

    # 使用x, y, model进行validate_times的交叉验证，返回精度序列(1*m)-ndarray
    def validate(self, x, y, bias=0):
        accuracy_list = []
        for rd in range(0, self.__validate_times):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.__test_size, random_state=rd+bias)
            self.__model.fit(x_train, y_train)
            # 选择精度计算的方式
            if self.__method == 'accuracy':         # 普通的分类精确度
                accuracy_list.append(self.__model.score(x_test, y_test))
            elif self.__method == 'weight_xu':     # 根据w-p权重的精确度提升
                tw = np.array([0.]*self.__kinds)
                for each_kind in range(0, self.__kinds):
                    tw[each_kind] = 1.0*len(np.where(y_test == each_kind+1)[0])
                tp = np.array([0.]*self.__kinds)
                y_predict = self.__model.predict(x_test)
                for i in range(0, len(y_test)):
                    if y_predict[i] == y_test[i]:
                        tp[int(y_test[i])-1] += 1
                tp /= tw
                tw /= tw.sum()
                accuracy_list.append(0)
                for i in range(0, len(tp)):
                    accuracy_list[rd] += (tp[i]-tw[i])*tw[i]
        accuracy_list = np.array(accuracy_list)
        return accuracy_list

    # 保存筛选后的输入集与输出集到本地
    def save(self):
        csvfile = open('final_xy.csv', 'wb')
        writer = csv.writer(csvfile)
        # 填充列名
        title = ['']*self.X_chosen.shape[1]
        tcd = np.hstack((0, self.X_check_days[:len(self.X_check_days)-1]+1))
        for i in range(1, len(tcd)):
            tcd[i] += tcd[i-1]
        for i in range(0, len(tcd)):
            string = "因子索引:"+str(self.X_choose_step[i])+" 处理:"+self.X_choose_method[i]
            title[tcd[i]] = string.decode('utf8').encode('gbk')
        writer.writerow(title)
        writer.writerows(self.X_chosen)
        csvfile.close()
