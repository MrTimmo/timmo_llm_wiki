import os
import numpy as np
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
import simple_mlp
import mlp_net
from mindspore import Model
from mindspore.nn import Accuracy

# 定义测试数据集，batch_size设置为1，则取出一张图片
ds_test = simple_mlp.create_dataset(os.path.join(simple_mlp.DATA_PATH, "test"), batch_size=1).create_dict_iterator()
data = next(ds_test)
# images为测试图片，labels为测试图片的实际分类
images = data["image"].asnumpy()
labels = data["label"].asnumpy()
# 加载已经保存的用于测试的模型
param_dict = load_checkpoint("checkpoint_lenet-10_938.ckpt")
# 加载参数到网络中
net = mlp_net.MLPNet()
load_param_into_net(net, param_dict)
# 使用函数model.predict预测image对应分类
model = Model(net)
output = model.predict(Tensor(data['image']))
predicted = output.asnumpy().argmax(axis=1)
print(f'Predicted: "{predicted[0]}", Actual: "{labels[0]}"')
