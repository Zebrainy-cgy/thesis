from mesh import *
from triangle import triangulate
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import tensor as Tensor
import matplotlib.pyplot as plt
from bf import *
from math import exp

deg = 2
Nloc = (deg + 1) * (deg + 2) // 2 # 单元上函数空间维数

print('local dimention:', Nloc)
eps = -1.0 # determine SIPG or NIPG
sigma = 10.0 * deg * deg # penalty variant

#################################################### set boundary condition ####################################################
# def source(realcor):
#     x = realcor[0]; y = realcor[1]
#     return 4 * (1-x**2-y**2) * exp(-x**2-y**2)

# def dirichlet(realcor):
#     x = realcor[0]; y = realcor[1]
#     return exp(-x**2-y**2)

def source(realcor):
    return 10

def dirichlet(realcor):
    return 0.0

#################################################### generate mesh ####################################################
v = [[0, 0], [0, 1], [1, 1], [1, 0]]
segments = [[0, 1], [1, 2], [2, 3], [3, 0]]

para = "pq30a0.1e"
print('Triangulate para', para)
t = triangulate({"vertices": v, 'segments': segments,}, para)
Mesh = t["triangles"]
points = Tensor(t["vertices"], dtype=torch.float32)
edges = t["edges"]
Nv = points.shape[0]
Nelt = Mesh.shape[0]


meshelt = []
for i in range(Nelt):
    tmpelt = element(Mesh[i])
    meshelt.append(tmpelt)

meshvertex = []
for i in range(Nv):
    meshvertex.append(points[i])

meshface = []
meshbdface = []
meshinface = []

Nface = edges.shape[0]
Nif = 0; Nbf = 0
for i in range(Nface):
    neighbor = [] # two cells alone the edge
    for k in range(Nelt):
        # assemble adjacency matrix and edge info
        if (edges[i][0] in Mesh[k]) and (edges[i][1] in Mesh[k]):
            neighbor.append(k)
    if len(neighbor) == 2:
        e = face(edges[i], neighbor)
        meshface.append(e)
        meshinface.append(e)
        Nif += 1
        # print('interior edge', i, j, 'is edge of cell', neighbor)
    elif len(neighbor) == 1:
        e = face(edges[i], neighbor)
        e.bctype = 1
        Nbf += 1
        meshface.append(e)
        meshbdface.append(e)

print('In the whole domain: ')
print(Nv, 'points')
print(Nelt, 'elements')
print(Nface, 'faces/edges')
print(Nif, 'interior edges,', Nbf, 'boundary edges')

print(meshvertex, Mesh)

#################################################### plot the mesh ##################################################
print("Plot the mesh:")
plt.triplot(points[:,0], points[:,1], Mesh) 
plt.plot(points[:,0], points[:,1], 'o') 
plt.axis('equal')
plt.savefig('mesh.png') 
plt.show() 


#################################################### define test functions, input: coordinates in reference cell ##################################################
phi = {0:phi0, 1:phi1, 2:phi2, 3:phi3, 4:phi4, 5:phi5}
grad_phi = {0:grad_phi0, 1:grad_phi1, 2:grad_phi2, 3:grad_phi3, 4:grad_phi4, 5:grad_phi5}


####################################################### u defined on reference element #########################################################
# class basisref(nn.Module):
#     # nn model based on reference cell
#     def __init__(self):
#         super(basisref, self).__init__()
#         self.fc1 = nn.Linear(2, 10)  
#         self.fc2 = nn.Linear(10, 10)
#         self.fc3 = nn.Linear(10, 1)
#     def forward(self, p):
#         '''p must be torch.tensor and requires_grad=True, dtype=torch.float32'''
#         op1 = F.gelu(self.fc1(p))
#         op2 = F.gelu(self.fc2(op1))
#         phi = self.fc3(op2)
#         return phi

class basisref(nn.Module):
    # nn model based on reference cell
    def __init__(self):
        super(basisref, self).__init__()
        self.fc1 = nn.Linear(2, 10)  
        self.fc2 = nn.Linear(10, 1)
    def forward(self, p):
        '''p must be torch.tensor and requires_grad=True, dtype=torch.float32'''
        op1 = F.gelu(self.fc1(p))
        phi = self.fc2(op1)
        return phi

# class basisref(nn.Module):
#     # nn model based on reference cell
#     def __init__(self):
#         super(basisref, self).__init__()
#         self.fc1 = nn.Linear(5, 10) 
#         self.fc2 = nn.Linear(10, 1)
#     def forward(self, p):
#         '''p must be torch.tensor and requires_grad=True, dtype=torch.float32'''
#         # print(p)
#         x = p[0:1]; y = p[1:2]
#         input = torch.stack([x, y, x*y, x**2, y**2], dim=-1)
#         o1 = self.fc1(input)
#         phi = self.fc2(o1)
#         return phi


if __name__ == "__main__":
    #################################################### test main ##########################################################
    # model = basisref()  
    
    # p = torch.tensor([0,0], requires_grad=True, dtype=torch.float32)  
    # output = model(p)  
    # print(output)
    
    # # 计算关于 p[0] 的偏导数  
    # output.backward() 
    # print(p.grad)
    pass