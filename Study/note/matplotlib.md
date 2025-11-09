# matplotlib 
 import matplotlib.pyplot as plt#绘图模块，导入pyplot包 设置别名plt  
 import nyumpy as np  

plt.plot()#线图和散点图  
plt.scatter()#散点图  
plt.bar()#bar"栅栏、横条”，垂直条形图和水平条形图  
plt.pie()#派，饼图  
plt.imshow()#图像，image  
plt.sublots()#创建子图  

 
## 绘制直线
 xpoints=np.arry([0,6])  
 ypoints=np.arry([0,100])  
 plt.plot(xpoints,ypoints)#绘制线图和散点图  
 plt.plot(xpoints,ypoints,'bo')  
 plt.show()  

## 绘制两个坐标点
import matplotlib.pyplot as plt  
import numpy as np  

xpoints=np.arry[1,8]  
ypoints=np.arry[3,5]  

plt.plot(xpoints,ypoints,'o')  
plt.show()  

## 绘制不规则线
import matplotlib.pyplot as plt  
import numpy as np  

xpoints=np.array([1,5,6,8])   
ypoints=np.array([4,5,6,7])  

plt.plot(xpoints,ypoints,'--ro')  
plt.show()  

## 为了显示中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]     

## 将图保存一个目录下
plt.savefig('results/wine_corr.png',dpi=300,bbox_inches='tight')  

## 显示图标
plt.bar(....,label='名字')  
plt.legend()