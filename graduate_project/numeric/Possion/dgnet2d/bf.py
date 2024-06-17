import torch  
import numpy as np  
  
# 判断点是否在参考单元内的函数  
def is_in_ref(p):  
    # 实现 is_in_ref 函数的代码  
    pass  
  
# 1  
def phi0(p):  
    if is_in_ref(p) != -1:  
        return torch.tensor(1.0, requires_grad=True, dtype=torch.float32)  
    else:  
        print('error') 
        return torch.tensor(0.0)  
  
# x  
def phi1(p):  
    if is_in_ref(p) != -1:  
        return p[0]  
    else:  
        print('error') 
        return torch.tensor(0.0)  
  
# y  
def phi2(p):  
    if is_in_ref(p) != -1:  
        return p[1]  
    else:  
        print('error') 
        return torch.tensor(0.0)  
  
# x^2  
def phi3(p):  
    if is_in_ref(p) != -1:  
        return p[0] * p[0]  
    else:  
        print('error') 
        return torch.tensor(0.0)  
  
# xy  
def phi4(p):  
    if is_in_ref(p) != -1:  
        return p[0] * p[1]  
    else:  
        print('error') 
        return torch.tensor(0.0)  
  
# y^2  
def phi5(p):  
    if is_in_ref(p) != -1:  
        return p[1] * p[1]  
    else:  
        print('error') 
        return torch.tensor(0.0)  
  
# (grad)basis functions  
# (0,0)  
def grad_phi0(p):  
    if is_in_ref(p) != -1:  
        return torch.tensor([0.0, 0.0])  
    else:  
        print('error')  
        return torch.zeros(2)  
  
# (1,0)  
def grad_phi1(p):  
    if is_in_ref(p) != -1:  
        return torch.tensor([1.0, 0.0])  
    else:  
        print('error')  
        return torch.zeros(2)  
  
# (0,1)  
def grad_phi2(p):  
    if is_in_ref(p) != -1:  
        return torch.tensor([0.0, 1.0])  
    else:  
        print('error')  
        return torch.zeros(2)  
  
# (2x,0)  
def grad_phi3(p):  
    if is_in_ref(p) != -1:  
        return torch.tensor([2 * p[0], 0.0])  
    else:  
        print('error')  
        return torch.zeros(2)  
  
# (x,y)  
def grad_phi4(p):  
    if is_in_ref(p) != -1:  
        return torch.tensor([p[1], p[0]])  
    else:  
        print('error')  
        return torch.zeros(2)  
  
# (0,2y)  
def grad_phi5(p):  
    if is_in_ref(p) != -1:  
        return torch.tensor([0.0, 2 * p[1]])  
    else:  
        print('error')  
        return torch.zeros(2)  