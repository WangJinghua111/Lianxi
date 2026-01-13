import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# flattern默认是True

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# 从训练数据中随机抽取10笔数据
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# np.random.choice(60000, 10)：从0到59999之间随机取10个数字
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 当监督数据是标签形式（非one_hot表示），交叉熵误差可以如下
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arrange(batch_size), t] + 1e-7)) / batch_size
