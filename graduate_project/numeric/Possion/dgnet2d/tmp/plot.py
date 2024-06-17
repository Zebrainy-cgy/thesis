import torch  
import torch.nn as nn  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np   
    
class basisref(nn.Module):
    # nn model based on reference cell
    def __init__(self):
        super(basisref, self).__init__()
        self.fc1 = nn.Linear(5, 10) 
        self.fc2 = nn.Linear(10, 1)
    def forward(self, p):
        '''p must be torch.tensor and requires_grad=True, dtype=torch.float32'''
        # print(p)
        x = p[0:1]; y = p[1:2]
        input = torch.stack([x, y, x*y, x**2, y**2], dim=-1)
        o1 = self.fc1(input)
        phi = self.fc2(o1)
        return phi


def plot():
    # 创建神经`网络实例  
    input_size = 2  
    hidden_size1 = 3  
    hidden_size2 = 4  
    output_size = 1  
    # model = basisref(input_size, hidden_size1, hidden_size2, output_size)  
    model = basisref()
    
    # 生成网格点  
    x_values = np.linspace(0, 1, 100)  
    y_values = np.linspace(0, 1, 100)  
    X, Y = np.meshgrid(x_values, y_values)  
    inputs = np.c_[X.ravel(), Y.ravel()]  
    # print(inputs, "X", X, "Y", Y)
    
    # 将输入转换为张量  
    inputs = torch.tensor(inputs, dtype=torch.float)  

    
    # 运行前向传播  
    outputs = np.zeros(inputs.shape[0])
    with torch.no_grad():  
        for i in range(inputs.shape[0]):
            outputs[i] = model(inputs[i])
    outputs = outputs.reshape(X.shape)  
    # 绘制3D图形  
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')  
    ax.plot_surface(X, Y, outputs, cmap='viridis')  
    
    ax.set_xlabel('Input 1')  
    ax.set_ylabel('Input 2')  
    ax.set_zlabel('Output')  
    ax.set_title('Neural Network Output')  
    
    plt.show() 

def ma():
    model = basisref()
    p = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float32)  
  
    # 前向传播  
    output = model(p)  
    gd = torch.autograd.grad(output, p, create_graph=True, retain_graph=True, allow_unused=True)[0]
    print(gd)
    print(output)

ma()
plot()