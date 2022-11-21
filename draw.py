import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


iteration1 = []
Loss1 = []
with open('C:\\Users\\Admin\\Desktop\\pygcn-master\\data\\train_acc.txt', 'r') as file:  # 打开文件
    for line in file.readlines():  # 文件内容分析成一个行的列表
        line = line.strip().split(" ")  # 按照空格进行切分
        itera, loss = line[2], line[7]  # 一行拆分为三行
        itera = int(itera)  # 保留itera参数
        iteration1.append(itera)  # 保存在数组中
        loss = float(loss)
        Loss1.append(loss)


# 画图
plt.title('Loss')  # 标题
# 常见线的属性有：color,label,linewidth,linestyle,marker等
plt.plot(iteration1, Loss1, color='cyan', label='loss')
#plt.legend()  # 显示上面的label
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()