import numpy as np

# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7 # 科学计数法，表示：1*10的-7次方
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0]
y = [0.1, 0.2, 0.6, 0.1]
cross_entropy_error(np.array(y),np.array(t))
# y + delta
# 列表 + 浮点数：X，在python中不被允许
# python只允许列表 + 列表
# np.array(y)把列表换成数组就可以运算了