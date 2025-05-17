import matplotlib.pyplot as plt
import csv

# 假设CSV文件有两列数据，第一列为X轴值，第二列为Y轴值
filename = 'data.csv'  # 替换为你的CSV文件名

# 读取CSV文件
with open(filename, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    x = []
    y = []
    for row in csvreader:
        if csvreader.line_num == 1:  # 跳过标题行
            continue
        x.append(int(row[0]))  # 假设第一列是X轴值
        y.append(int(row[1]))  # 假设第二列是Y轴值

# 绘制折线图
plt.plot(x, y, label='CSV Data')

# 添加图表标题和坐标轴标签
plt.title('CSV Line Graph')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
