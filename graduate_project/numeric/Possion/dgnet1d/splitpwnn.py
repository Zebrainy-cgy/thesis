import torch
import torch.nn as nn
from torch import tensor as Tensor
import matplotlib.pyplot as plt
from math import exp
import numpy as np

print("This method is based on Particle WNN, but split the net into N nets!")
N = 10
Nint = 50
h = 1.0 / N
eps = -1.0
deg = 2
sigma = 10.0 * deg * deg
X = torch.linspace(0, 1, N+1)
print("The x is: ", X)

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

def source(x):
    return 10

def exact(x):
    return (1-x)*torch.exp(-x**2)

###################################################### Particle test function ##########################################################
def particletest(r):
    '''particle test function in the nth cell, the refenece coordinate is x'''
    if r <= 1+1e-6 and r>-1e-6:
        return (1-r)**5 * (8*r**2+5*r+1)
    else:
        print("r>1, somthing went wrong!")


def grad_particletest(r):
    if r <= 1+1e-6 and r>-1e-6:
        return -14*r*(4*r+1)*(1-r)**4
    else:
        print("r>1, somthing went wrong!")

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

############################################## define nn on the interval ##################################################
class u(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)
    def forward(self, x):
        x = x[0:1]
        input = torch.cat([x, x**2])
        o1 = self.layer1(input)
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


def computeP(bflist):
    particleloss = Tensor([0.0])
    for n in range(N):
        P = Tensor([0.0])
        hn = X[n+1] - X[n]
        bf = bflist[n]
        s = np.linspace(X[n], X[n+1], Nint, endpoint=False)
        xm = (X[n+1] + X[n]) / 2.0
        for k in range(Nint-2):
            p = Tensor([s[k]], requires_grad=True, dtype=torch.float32)
            up = bf(p)
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

def computebd(bflist):
    bf1 = bflist[0]; bf2 = bflist[N-1]
    # bdloss = (bf1(Tensor([X[0]], requires_grad=True))- Tensor([1.0]))**2
    bdloss = (bf1(Tensor([X[0]], requires_grad=True)))**2
    bdloss += (bf2(Tensor([X[N]], requires_grad=True)))**2
    return bdloss

def computeCon(bflist):
    con = Tensor([0.0])

    for n in range(1, N):
        bfl = bflist[n - 1]; bfr = bflist[n]
        p = Tensor([X[n]], requires_grad=True)
        ul = bfl(p); ur = bfr(p)

        con += (ul - ur)**2
        # print((ul - ur)**2)
    return con

def computeGradCon(bflist):
    gcon = Tensor([0.0])

    for n in range(1, N):
        bfl = bflist[n - 1]; bfr = bflist[n]
        p = Tensor([X[n]], requires_grad=True)
        ul = bfl(p); ur = bfr(p)
        gul = torch.autograd.grad(ul, p, retain_graph=True, create_graph=True)[0]
        gur = torch.autograd.grad(ur, p, retain_graph=True, create_graph=True)[0]
        gcon += (gul - gur)**2
    return gcon

class Solver:
    def __init__(self) -> None:
        bflist = []
        for _ in range(N):
            bf = u()
            bflist.append(bf)
        self.models = bflist
        self.parameters = aggpara(self.models)
        print("The optimizer is LBFGS.")
        self.optimizer = torch.optim.LBFGS(self.parameters, lr=1.0, max_iter=200, max_eval=50000, history_size=50, 
                                           tolerance_grad=1e-7, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn='strong_wolfe')
        self.maxIter = 1000
        self.iter = 1
    
    def loss(self):
        ########################### particle loss ###############################
        self.optimizer.zero_grad()
        particleloss = 2.0 * computeP(self.models) / N
        conloss = computeCon(self.models) 
        bdloss = computebd(self.models) / 2
        gconloss = computeGradCon(self.models) 

        loss = particleloss + bdloss + conloss + gconloss
        loss.backward()
        self.iter += 1
        if self.iter % 10 == 0:
            print(f"The {self.iter}th training, loss is {loss.item()}: paricleloss is {particleloss.item()}, conloss is {conloss.item()}, bdloss is {bdloss.item()}, gconloss is {gconloss.item()}")
        return loss
    
    def train(self):
        trainplug(self.models)
        self.optimizer.step(self.loss)
        loss = self.loss()
        print(f"Finish training! The {self.iter}th training loss is {loss.item()}")
        


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
net.train()


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


print("################################ Plot the Net solution #################################")
XX = torch.tensor([])  
YY = torch.tensor([]) 
for i in range(N):
    xreal, y = sol(i)
    XX = torch.cat((XX, xreal), dim=0)
    YY = torch.cat((YY, y), dim=0)
XX = XX.detach().numpy()
YY = YY.detach().numpy()
plt.plot(XX, YY, label='net')
plt.xlabel('x') 
plt.ylabel('y') 

################################# plot test function ###################################
# for notf in range(3):
#     TF = torch.tensor([]) 
#     for i in range(N):
#         _, y = tforigin(notf, i)
#         TF = torch.cat((TF, y), dim=0)
#     TF = TF.detach().numpy()
#     plt.plot(XX, TF, label=f'tf{notf}')
    

print("################################# plot exact solution ###################################")
# x_values = np.linspace(0, 1, 1000)  # 生成 x 值  
# y_values = (1 - x_values) * np.exp(-x_values**2)  # 计算对应的 y 值  
# plt.plot(x_values, y_values, label='(1-x)e^x)')  

plt.xlabel('x')  
plt.ylabel('y')  
plt.legend() 
plt.show() 