import torch
import torch.nn as nn
import numpy as np
from torch import tensor as Tensor
from collections import OrderedDict
import matplotlib.pyplot as plt

# def source(x):
#     return (4*x**3 - 4*x**2-6*x + 2) * torch.exp(-x**2)
def source(x):
    return x**0 * 10

def exact(x):
    return (1-x)*torch.exp(-x**2)

class net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, act=torch.nn.Tanh) -> None:
        super(net, self).__init__()
        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))
        for i in range(depth):
            layers.append(('%d hidden_layer' %i, torch.nn.Linear(hidden_size, hidden_size)))
            layers.append(('%d hidden_activation' %i, act()))
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))
        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        output = self.layers(x)
        return output
class Solver:
    def __init__(self) -> None:
        self.model = net(input_size=1, hidden_size=20, output_size=1, depth=4)
        self.N = 10
        self.h = 1.0 / 10
        self.x = torch.linspace(0,1,self.N+1).unsqueeze(1)
        self.x.requires_grad = True
        # print(self.x.shape)
        lb = Tensor([0.0])
        rb = Tensor([1.0])
        self.bd =torch.cat((lb, rb)).unsqueeze(1)
        self.bdv = Tensor([[0.0], [0.0]])
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=50000, max_eval=50000, history_size=50, 
                                           tolerance_grad=1e-7, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn='strong_wolfe')
        
    def lossfunc(self):
        self.optimizer.zero_grad()
        y_bd = self.model(self.bd)
        y_pred = self.model(self.x)

        grad_u = torch.autograd.grad(inputs=self.x, outputs=y_pred, grad_outputs=torch.ones_like(y_pred),
                                     create_graph=True, retain_graph=True)[0]
        # print(grad_u)

        u_xx = torch.autograd.grad(inputs=self.x, outputs=grad_u, grad_outputs=torch.ones_like(grad_u),
                                     create_graph=True, retain_graph=True)[0]
        # print(u_xx.shape)
        f = source(self.x)
        loss_pde = self.criterion(-u_xx, f)
        loss_data = self.criterion(y_bd, self.bdv)

        loss = 5 * loss_data + loss_pde
        
        if self.iter % 100 == 0:
            print(self.iter, loss.item())
        self.iter += 1
        loss.backward()
        return loss
    
    def train(self):
        self.model.train()
        self.optimizer.step(self.lossfunc)
        loss = self.lossfunc()
        print(f"Finish training! The {self.iter}th training loss is {loss.item()}")

model = Solver()
model.train()

x_values = np.linspace(0, 1, 1000)  # 生成 x 值  
y_values = (1 - x_values) * np.exp(-x_values**2)  # 计算对应的 y 值  
plt.plot(x_values, y_values, label='(1-x)e^x)')  

x = torch.linspace(0,1,1000).unsqueeze(1)  # 生成 x 值 
y = model.model(x).detach().numpy()

plt.plot(x,y)
plt.xlabel('x')  
plt.ylabel('y')  
plt.show() 