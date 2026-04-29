# 导入需要用到的模块
import mindspore.nn as nn
# 定义线性模型
class MLPNet(nn.Cell):
    def __init__(self):
        super(MLPNet, self).__init__()
        # 定义展平层
        self.flatten = nn.Flatten()
        # 输入层
        self.dense1 = nn.Dense(32*32, 128)
        # hiden1
        self.dense2 = nn.Dense(128, 64)
        # 输出层
        self.dense3 = nn.Dense(64, 10)

    # 等价于forward，定义前向传播方法
    def construct(self, inputs):
        # 1. 第一步展平
        x = self.flatten(inputs)
        # 2. 第2、3、4全连接神经网络，返回应该会是一个10维的向量
        x = nn.ReLU()(self.dense1(x))
        x = nn.ReLU()(self.dense2(x))
        logits = self.dense3(x)
        return logits