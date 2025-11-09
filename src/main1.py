import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import seaborn as sns
import pandas as pd
import numpy as np
import timeit
from sklearn.datasets import load_iris
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"] 

#iris的数据集，z_scores标准化,绘制均值方差图，相关性热力图，z = (x - μ) / σ

#利用python进行z_score
data_python=load_iris().data.tolist()
data_python_z_score=[]
for x in range(len(data_python)):
    data_python_z_score.append([0.00 ,0.00 ,0.00 ,0.00])
for j in range(0,4,1):
    
    sum_col=0
    sum_s=0
    for i in range(len(data_python)):
        sum_col+=data_python[i][j]
    
    x_mean=sum_col/len(data_python)
    for i in range(len(data_python)):
        sum_s+=((data_python[i][j]-x_mean)**2)
    x_std=(sum_s/len(data_python))**0.5
    for i in range(len(data_python)):
        data_python_z_score[i][j]=(data_python[i][j]-x_mean)/x_std 

#利用numpy向量化进行z_score
data_numpy=load_iris().data
X_mean=data_numpy.mean(axis=0)
X_std=data_numpy.std(axis=0)
data_numpy_z_score=(data_numpy-X_mean)/X_std

#计算时间比值
python_z_code='''
data_python=load_iris().data.tolist()
data_python_z_score=[]
for x in range(len(data_python)):
    data_python_z_score.append([0.00 ,0.00 ,0.00 ,0.00])
for j in range(0,4,1):
    
    sum_col=0
    sum_s=0
    for i in range(len(data_python)):
        sum_col+=data_python[i][j]
    
    x_mean=sum_col/len(data_python)
    for i in range(len(data_python)):
        sum_s+=((data_python[i][j]-x_mean)**2)
    x_std=(sum_s/len(data_python))**0.5
    for i in range(len(data_python)):
        data_python_z_score[i][j]=(data_python[i][j]-x_mean)/x_std 
'''
python_z_score_time=timeit.timeit(
python_z_code,
number=1000,
globals=globals()
)
numpy_z_code='''
X_mean=data_numpy.mean(axis=0)
X_std=data_numpy.std(axis=0)
data_numpy_z_score=(data_numpy-X_mean)/X_std
'''
numpy_z_score_time=timeit.timeit(
numpy_z_code,
number=1000,
globals=globals()
)
time_contrast=python_z_score_time/numpy_z_score_time
print(f"python利用for所需时间为：{python_z_score_time:.2f}秒")
print(f"numpy向量化所需时间为：{numpy_z_score_time:.2f}秒")
print(f"pyhton消耗的时间是numpy消耗时间的{time_contrast:.2f}倍")#
#发现对pthon_code的处理对倍数影响很大，特别是加载数据步骤

#wine的数据集，z_scores标准化,绘制均值方差图，相关性热力图
from sklearn.datasets import load_wine
data_wine=load_wine().data
data_wine_mean=data_wine.mean(axis=0)
data_wine_std=data_wine.std(axis=0)
data_wine_z_score=(data_wine-data_wine_mean)/data_wine_std

feature_names=load_wine().feature_names
data_wine_frame=pd.DataFrame(data=data_wine_z_score,columns=feature_names)
data_wine_corr=data_wine_frame.corr(method='pearson')

##均值方差图
plt.figure(figsize=(16,8))
width=0.5
x = np.arange(len(feature_names))
x_mean=x-width/2
x_std=x+width/2
plt.bar(x_mean,data_wine_mean,width,color='skyblue',label='均值')
plt.bar(x_std,data_wine_std,width,color='orange',label="方差")
plt.tight_layout
plt.xticks(
    ticks=x,
    labels=feature_names,#把x轴替换回来
    rotation=45,
    ha='right',
    fontsize=10
)
plt.title('wine的特征均值图') 
plt.legend()
plt.savefig('results/wine_meanstd.png',dpi=300,bbox_inches='tight')
plt.show()

##热力图
plt.figure(figsize=(14,8))
sns.heatmap(data_wine_corr,annot=True,cmap='coolwarm',fmt='.2f')
plt.tight_layout
plt.title('wine的热力相关图')
plt.savefig('results/wine_corr.png',dpi=300,bbox_inches='tight')
plt.show()

#拆分写配置之类的