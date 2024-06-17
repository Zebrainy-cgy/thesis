import torch
import torch.nn as nn
from torch import tensor as Tensor
import matplotlib.pyplot as plt
from math import exp
import numpy as np

N = 10
h = 1.0 / N
eps = -1.0
deg = 2
sigma = 10.0 * deg * deg
X = torch.linspace(0, 1, N+1)
Nint = 50

# def source(x):
#     return (4*x**3 - 4*x**2-6*x + 2) * torch.exp(-x**2)

def source(x):
    return 10

def exact(x):
    return (1-x)*torch.exp(-x**2)

###################################################### Particle test function ##########################################################
def particletest(r):
    '''particle test function in the nth cell, 0<r<1'''
    if r <= 1+1e-6 and r>-1e-6:
        return (1-r)**5 * (8*r**2+5*r+1)
    else:
        print("r>1, somthing went wrong!")
        return 0


def grad_particletest(r):
    ''' gradient of test function on the reference cell, only accept r'''
    if r <= 1+1e-6 and r>-1e-6:
        return -14*r*(4*r+1)*(1-r)**4
    else:
        print(r, "r>1, somthing went wrong!")
        return 0

############################################## plot the particle test function ##################################################
def plot_testfunc():
    x1 = torch.linspace(0,1,100)
    x2 = torch.linspace(-1,0,100)
    r = -x2
    y1 = torch.zeros_like(x1)
    y2 = torch.zeros_like(x2)
    gy1 = torch.zeros_like(x1)
    gy2 = torch.zeros_like(x2)
    for i in range(x1.shape[0]):
        y1[i] = particletest(x1[i])
        y2[i] = particletest(r[i])
        gy1[i] = grad_particletest(x1[i])
        gy2[i] = -grad_particletest(r[i])

    plt.plot(x1,y1)
    plt.plot(x2, y2)
    plt.plot(x1,gy1)
    plt.plot(x2, gy2)
    plt.show()

class net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(1, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 1)
    def forward(self, x):
        o1 = torch.tanh(self.layer1(x))
        o2 = torch.tanh(self.layer2(o1))
        op = self.layer3(o2)
        return op



def ref2phi(s, n):
    ''' s is the position of one dim'''
    return (s + 1) / 2.0 * (X[n+1] - X[n]) + X[n]

def phi2ref(x, n):
    xm = (X[n] + X[n+1])/2.0
    return abs(x - xm)/(- X[n] + X[n+1])*2.0 



def computeP(net: net):
    particleloss = Tensor([0.0])
    for n in range(N):
        P = Tensor([0.0])
        hn = X[n+1] - X[n]
        
        s = np.linspace(X[n], X[n+1], Nint, endpoint=False)
        xm = (X[n+1] + X[n]) / 2.0
        for k in range(Nint-2):
            p = Tensor([s[k]], requires_grad=True, dtype=torch.float32)
            up = net(p)
            gup = torch.autograd.grad(up, p, create_graph=True, retain_graph=True)[0]
            r = phi2ref(s[k], n)
            # print(k, s[k], 'r', r)
            gptf = grad_particletest(r)
            if p - xm < 0:
                gr = -2.0 / hn
            else:
                gr = 2.0 / hn
            f = source(p)
            ptf = particletest(r)
            P += 1.0 / Nint * (gup * gptf * gr - f * ptf) * hn
        particleloss += P**2
    return particleloss



def computebd(net: net):
    # bdloss = (net(Tensor([X[0]]))- Tensor([1.0]))**2
    bdloss = (net(Tensor([X[0]])))**2
    bdloss += (net(Tensor([X[N]])))**2
    return bdloss


class Solver:
    def __init__(self) -> None:
        self.model = net()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=50000, max_eval=50000, history_size=50, 
                                           tolerance_grad=1e-7, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn='strong_wolfe')
        self.maxIter = 1000
        self.iter = 1
    
    def loss(self):
        self.optimizer.zero_grad()
        particleloss = computeP(self.model) / (N)
        bdloss = 5.0 * computebd(self.model) / 2
        loss = particleloss + bdloss 
        
        self.iter += 1
        if self.iter % 10 == 0:
            print(f"The {self.iter}th training, loss is {loss.item()}: paricleloss is {particleloss.item()},  bdloss is {bdloss.item()}")
        loss.backward()
        return loss
    
    def train(self):
        self.model.train()
        self.optimizer.step(self.loss)
        loss = self.loss()
        print(f"Finish training! The {self.iter}th training loss is {loss.item()}")
        

net = Solver()
net.train()


bfl = net.model
    

print("################################# plot exact solution ###################################")
# x_values = np.linspace(0, 1, 1000)  # 生成 x 值  
# y_values = (1 - x_values) * np.exp(-x_values**2)  # 计算对应的 y 值  
# plt.plot(x_values, y_values, label='(1-x)e^x)')  

print("################################ Plot the Net solution #################################")
x = np.linspace(0, 1, 1000)  # 生成 x 值 
y = np.zeros_like(x)
for i in range(x.shape[0]):
    p = Tensor([x[i]], requires_grad=True, dtype=torch.float32)
    y[i] = bfl(p)  
plt.plot(x,y)
plt.xlabel('x')  
plt.ylabel('y')  
plt.show() 