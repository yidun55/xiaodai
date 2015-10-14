#coding: utf-8
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors
from pyspark.mllib.classification import SVMWithSGD, SVMModel
import math


#加载数据并去除某些行
def load_rm(line, col_rm):
    """
    col_rm:要去除的列(从零开始计数)
    """
    tmp = line.strip().split(",")
    features = [tmp[i] for i in xrange(len(tmp)) if i not in col_rm]
    features = ["0" if ele=="" else ele for ele in features]
    return LabeledPoint(tmp[1], features)


def data_normalization(p, columns):
    """
    对数据进行归一化，如反正切，标准化等处理
    """
    log10_trans_col = [12, 13, 15, 16, 25, 27]
    normal_features = []
    for i in xrange(columns):
        # tmp = normal_map[i]
        # normal_features.append((p.features[i]-tmp[1])/(tmp[0]-tmp[1]))
        #归一化处理的公式new_x = (x-min)/(max-min)
        # normal_features.append(math.log10(p.features[i]+1))
        #归一化的公式 new_x = log10(x+1)
        # normal_features.append(math.log10((p.features[i]-tmp[1])/(tmp[0]-tmp[1])))
        #归一化的公式new_x = log10((x-min)/(max-min))
        # if i in log10_trans_col: #对12,13,15,16,25,27进行log10(x)/log10(max)转换
        #     normal_features.append((math.log10(p.features[i]+1))/math.log10(tmp[1]))
        # else:
        #     normal_features.append(p.features[i])

        tmp1 = std_map[i]
        normal_features.append((p.features[i]-tmp1[0])/tmp1[1])
        #进行标准化处理new_x = (x-mean)/std
        # normal_features.append(p.features[i]-tmp1[0])
        # 进行new_x = x - mean
        # normal_features.append(math.atan(p.features[i])*2/math.pi)
        #反正切转换new_x=atan(x)*w/pi
    return LabeledPoint(p.label, normal_features)


sc = SparkContext(appName='xiaodai')

data_path = "/data/chengqj/myxiaodai.csv"
#col_remove = [0,1,2,4,25,26] + range(19,24) + range(28, 36)  #去除所有离散变量
col_remove = [0,1,2,4,25,26] + range(19,24) #

data = sc.textFile(data_path)
pdata = data.map(lambda line: load_rm(line, col_remove))


#计算各个非离散变量的最大值和最小值，并以字典的形式保存
#其形式如下
f_length = len(list(pdata.first().features))
normal_map = {}
for i in xrange(f_length):
    max_v = pdata.map(lambda p: p.features[i]).max()
    min_v = pdata.map(lambda p: p.features[i]).min()
    normal_map[i] = (max_v, min_v)

std_map = {}
for i in xrange(f_length):
    mean_v = pdata.map(lambda p: p.features[i]).mean()
    std_v = math.sqrt(pdata.map(lambda p: p.features[i]).variance())
    std_map[i] = (mean_v, std_v)


#数据处理
# resample_time = pdata.filter(lambda p:p.label==0).count()/float(pdata.filter(lambda p:p.label==1).count())
# pdata = pdata.filter(lambda p:p.label==0).union(pdata.filter(lambda p:p.label==1).sample(true, resample_time))
normal_data = pdata.map(lambda p: data_normalization(p, f_length))


#切分训练集和测试集
(trianData, testData) = pdata.randomSplit([0.7, 0.3], seed=1)


#训练模型
inters = 100
reParam = 0
model = SVMWithSGD.train(trianData, iterations = inters, \
    regParam=reParam, intercept=False)


#评估模型
LAP = testData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = LAP.filter(lambda (v, p):v!=p).count()/float(LAP.count())
precision = LAP.filter(lambda (v,p):v==p==1).count()/float(LAP.filter(lambda (v,p):p==1).count())
recall = LAP.filter(lambda (v, p): v==p==1).count()/float(LAP.filter(lambda (v,p):v==1).count())
print("training Error = " + str(trainErr))
print("TP/(TP+FP)" + str(precision) + "##########################")
print("TP/(TP+FN)"+ str(recall) + "@@@@@@@@@@@@@@@@@@@@@@@@@")