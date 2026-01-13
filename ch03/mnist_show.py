import sys, os
sys.path.append(os.pardir) # 为了导入父目录中的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 调用
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# flatten：把图像展平成一维向量（本来是28*28，现在是1*784）
# one_hot_label：标签是[0, 1, 0, 0]，即正确解标签是1，其余是0
# one_hot_label是False，只像7，2这样简单保存标签
img = x_train[0]
label = t_train[0]
print(label) # 5

print(img.shape) # (784,)
img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
print(img.shape) # (28, 28)

img_show(img)

# 输出各个数据的形状
print(x_train.shape) # (60000, 784) 每列代表一张图，有60000张图
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)