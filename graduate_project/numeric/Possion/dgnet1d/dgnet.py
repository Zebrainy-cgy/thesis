import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor as Tensor
import matplotlib.pyplot as plt
from math import exp, pi,cos, sin
import numpy as np
import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

N = 20
Nint = 50
h = 1.0 / N
eps = -1.0
deg = 2
X = torch.linspace(0, 1, N+1).to(device)
print(X, X.dtype)

def testfunc(x, i, n):
    ''' ith test function of cell n'''
    x_mid = (X[n] + X[n+1]) / 2.0
    return 2**i * (x - x_mid)**i / (X[n+1] - X[n])** i

def grad_testfunc(x, i, n):
    ''' gradience of ith test function of cell n'''
    if i == 0:
        return 0
    else:
        x_mid = (X[n] + X[n+1]) / 2.0
        return 2**i * i * (x - x_mid)**(i-1) / (X[n+1] - X[n])** i


# def source(x):
#     return (4*x**3 - 4*x**2-6*x + 2) * exp(-x**2)

w = 15*pi

def source(w, x):
    return 2*w*sin(w*x)+w**2*x*cos(w*x)

# def exact(x):
#     return (1-x)*torch.exp(-x**2)

def exact(w , x):
    return x*torch.cos(w * x)

# class u(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.layer1 = nn.Linear(2, 10)
#         self.layer2 = nn.Linear(10, 1)
#     def forward(self, x):
#         x = x[0:1]
#         input = torch.cat([x, x**2]).to(device)
#         o1 = F.gelu(self.layer1(input))
#         op = self.layer2(o1)
#         print(op.device)
#         return op

class u(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)
    def forward(self, x):
        o1 = F.gelu(self.layer1(x))
        op = self.layer2(o1)
        return op

bflist = []
for _ in range(N):
    bf = u()
    bflist.append(bf)


def ref2phi(s, n):
    ''' s is the position of one dim'''
    return (s + 1) / 2.0 * (X[n+1] - X[n]) + X[n]

def phi2ref(x, n):
    xm = (X[n] + X[n+1])/2.0
    return abs(x - xm)/(- X[n] + X[n+1])*2.0 

def computeCell(bflist, n, notf):
    ''' nth cell, ith test function'''
    hn = X[n+1] - X[n]
    Cell = Tensor([0.0]).to(device)
    bf = bflist[n]
    s = torch.linspace(X[n], X[n+1], Nint)
    for k in range(Nint):
        p = Tensor([s[k]], requires_grad=True, dtype=torch.float32).to(device)
        up = bf(p)
        gup = torch.autograd.grad(up, p, create_graph=True, retain_graph=True)[0]
        
        v = testfunc(p, notf, n)
        gv = grad_testfunc(p, notf, n)

        f = source(w, p)

        Cell += (gup * gv - f * v) * hn * 1.0 / Nint
        
    lp = Tensor([X[n]], requires_grad=True, dtype=torch.float32).to(device)
    rp = Tensor([X[n+1]], requires_grad=True, dtype=torch.float32).to(device)
    ul = bf(lp); ur = bf(rp)
    gul = torch.autograd.grad(ul, lp, create_graph=True, retain_graph=True)[0]
    gur = torch.autograd.grad(ur, rp, create_graph=True, retain_graph=True)[0]
    vl = testfunc(lp, notf, n)
    vr = testfunc(rp, notf, n)
    Cell += gul * vl - gur * vr
    return Cell

def computebd(bflist):
    bf1 = bflist[0]; bf2 = bflist[N-1]
    # bdloss = (bf1(Tensor([X[0]], requires_grad=True))- Tensor([1.0]))**2
    pinit = Tensor([X[0]], requires_grad=True).to(device)
    pfinal = Tensor([X[N]], requires_grad=True).to(device)
    bdloss = (bf1(pinit)-exact(w, pinit))**2
    bdloss += (bf2(pfinal) - exact(w, pfinal))**2
    
    return bdloss

def computeCon(bflist):
    con = Tensor([0.0]).to(device)

    for n in range(1, N):
        bfl = bflist[n - 1]; bfr = bflist[n]
        p = Tensor([X[n]], requires_grad=True).to(device)
        ul = bfl(p); ur = bfr(p)

        con += (ul - ur)**2
        
    return con

def computeGradCon(bflist):
    gcon = Tensor([0.0]).to(device)

    for n in range(1, N):
        bfl = bflist[n - 1]; bfr = bflist[n]
        p = Tensor([X[n]], requires_grad=True).to(device)
        ul = bfl(p); ur = bfr(p)
        gul = torch.autograd.grad(ul, p, retain_graph=True, create_graph=True)[0]
        gur = torch.autograd.grad(ur, p, retain_graph=True, create_graph=True)[0]
        gcon += (gul - gur)**2
    return gcon

class Solver:
    def __init__(self) -> None:
         
        bflist = []
        for _ in range(N):
            bf = u().to(device)
            bflist.append(bf)
        self.models = bflist
        self.parameters = aggpara(self.models)
        self.optimizer = torch.optim.LBFGS(self.parameters, lr=1.0, max_iter=300, max_eval=50000, history_size=50, 
                                           tolerance_grad=1e-7, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn='strong_wolfe')
        self.maxIter = 1000
        self.iter = 1
    
    def loss(self):
        self.optimizer.zero_grad()
        ########################### dgloss ####################################
        tfloss = torch.zeros((deg + 1, 1)).to(device)
        for i in range(deg + 1):
            for k in range(N):
                A = computeCell(self.models, k, i)
                tfloss[i] += A ** 2
        dgloss = torch.sum(tfloss).unsqueeze(0)
        loss = dgloss

        conloss = computeCon(self.models) 
        bdloss = computebd(self.models) / 2
        gconloss = computeGradCon(self.models) 

        loss += bdloss + conloss + gconloss

        loss.backward()
        self.iter += 1
        if self.iter % 10 == 0:
            print(f"The {self.iter}th training, loss is {loss.item()}: dgloss is {dgloss.item()}, conloss is {conloss.item()}, bdloss is {bdloss.item()}, gconloss is {gconloss.item()}")
        
        return loss

    
    def train(self):
        trainplug(self.models)
        self.optimizer.step(self.loss)
        loss = self.loss()
        for model in self.models:  
            model.to('cpu')
        print(f"Finish training! The {self.iter}th training, loss is {loss.item()}!")
        


def aggpara(bflist):
    parameters = list()
    for i in range(N):
        parameters += list(bflist[i].parameters())
    return parameters
def trainplug(bflist):
    for i in range(N):
        bflist[i].train()
def evalplug(bflist):
    for i in range(N):
        bflist[i].eval()
    
net = Solver()
# net.train()

bfl = net.models

def sol(n):
    x = torch.linspace(-1, 1, 100)[1:-1]
    xreal = torch.zeros_like(x)
    y = torch.zeros_like(x)
    for i in range(x.shape[0]):
        xreal[i] = ref2phi(x[i], n)
        y[i] = bfl[n](Tensor([xreal[i]]))
    return xreal, y

def tforigin(notf, n):
    x = torch.linspace(-1, 1, 100)[1:-1]
    xreal = torch.zeros_like(x)
    y = torch.zeros_like(x)
    for i in range(x.shape[0]):
        xreal[i] = ref2phi(x[i], n)
        y[i] = testfunc(Tensor([xreal[i]]), notf , n)
    return xreal, y

################################ plot the solution ####################################
# XX = torch.tensor([])  
# YY = torch.tensor([]) 
# for i in range(N):
#     xreal, y = sol(i)
#     XX = torch.cat((XX, xreal), dim=0)
#     YY = torch.cat((YY, y), dim=0)
# XX = XX.detach().numpy()
# YY = YY.detach().numpy()
# plt.plot(XX, YY, label='net')
# plt.xlabel('x') 
# plt.ylabel('y') 

################################# plot test function ###################################
# for notf in range(3):
#     TF = torch.tensor([]) 
#     for i in range(N):
#         _, y = tforigin(notf, i)
#         TF = torch.cat((TF, y), dim=0)
#     TF = TF.detach().numpy()
#     plt.plot(XX, TF, label=f'tf{notf}')
    

################################# plot exact solution ###################################
x_values = torch.linspace(0, 1, 1000)  # 生成 x 值  
# y_values = (1 - x_values) * np.exp(-x_values**2)  # 计算对应的 y 值  
y_values = exact(w, x_values)
plt.plot(x_values, y_values)  


plt.xlabel('x')  
plt.ylabel('y')  
plt.legend() 
plt.show() 