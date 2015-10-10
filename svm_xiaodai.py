#coding: utf-8
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors
from pyspark.mllib.classification import SVMWithSGD, SVMModel


#加载数据并去除某些行
def load_rm(line, col_rm):
    """
    col_rm:要去除的列(从零开始计数)
    """
    tmp = line.strip().split(",")
    features = [tmp[i] for i in xrange(len(tmp)) if i not in col_rm]
    features = ["0" if ele=="" else ele for ele in features]
    return LabeledPoint(tmp[1], features)

data_path = "/data/chengqj/myxiaodai.csv"
col_remove = [0,1,2,4,25,26] + range(19,24) + range(28, 36)

sc = SparkContext(appName='xiaodai')
data = sc.textFile(data_path)
pdata = data.map(lambda line: load_rm(line, col_remove))


#切分训练集和测试集
(trianData, testData) = pdata.randomSplit([0.7, 0.3])

#训练模型
inters = 100
model = SVMWithSGD.train(trianData, iterations = inters)

#评估模型
LAP = testData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = LAP.filter(lambda (v, p):v!=p).count()/float(LAP.count())
print("training Error = " + str(trainErr))
