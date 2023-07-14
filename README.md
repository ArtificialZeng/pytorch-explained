# pytorch-explained
pytorch-explained annatated




import torch.nn as nn # 导入PyTorch的神经网络模块，并以"nn"作为别名
import torch.nn.functional as F # 导入PyTorch的函数接口模块，并用"F"作为别名

class Model(nn.Module): # 定义一个名为Model的新类，它继承自nn.Module
    def __init__(self): # 定义类的构造函数
        super().__init__() # 调用父类nn.Module的初始化方法，为PyTorch的内部机制设置基础状态
        self.conv1 = nn.Conv2d(1, 20, 5) # 定义一个二维卷积层，并赋值给self.conv1
        self.conv2 = nn.Conv2d(20, 20, 5) # 定义第二个二维卷积层，并赋值给self.conv2

    def forward(self, x): # 定义前向传播函数
        x = F.relu(self.conv1(x)) # 将输入通过第一卷积层，然后应用ReLU激活函数
        return F.relu(self.conv2(x)) # 将输出通过第二卷积层，然后应用ReLU激活函数并返回结果

