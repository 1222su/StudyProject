###配置是啥
###数据集内容
###日志是怎么写的
###timeit是啥
###预热并清理缓存是啥
###路径是怎么保存的
import os
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.datasets import load_iris
iris=load_iris()
data=iris.data
print(iris)
