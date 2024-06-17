import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from triangle import triangulate
from mesh import *
from bf import *
from math import sin, pi, cos

deg = 2
Nloc = (deg + 1) * (deg + 2) // 2 

eps = 1.0 # determine SIPG or NIPG
sigma =  10.0 * deg *deg # penalty variant


######################################################################################
# set the equation
def source(p):
    x = p[0]; t = p[1]
    return -pi * sin(pi * t)* sin(pi * x)+pi * pi *cos(pi*t)* sin(pi * x)
# def source(p):
#     return 1

def initu(p):
    return sin(p[0]* pi)

def dirichlet(p):
    return 0

def exact(p):
    x = p[0]; t = p[1]
    return sin(pi * x)*cos(pi*t)

# def source(p):
#     x = p[0]; t = p[1]
#     return 0.5*(t-0.5)**(x*2+(t-0.5)**2)**(-0.75)+(-0.5+0.75*x**2)*(x**2+(t-0.5)**2)**(-1.75)

# def initu(p):
#     x = p[0]+1e-4
#     return (x**2+0.25)**(0.25)

# def dirichlet(p):
#     x = p[0]+1e-4; t = p[1]+1e-4
#     return (x**2+(t-0.5)**2)**(0.25)


######################################################################################
# generate mesh with Triangle
# v = [[0, 0], [1, 0], [1, 1], [0, 1], 
#      [0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6], 
#      [0.3, 0.3], [0.7, 0.3],[0.7, 0.7],[0.3, 0.7],
#      [0.2, 0.2],[0.8, 0.2],[0.8, 0.8],[0.2, 0.8],
#      [0.1, 0.1],[0.9, 0.1],[0.9, 0.9],[0.1, 0.9]]
# segments = [[0, 1], [1, 2], [2, 3], [3, 0], 
#             [4,5],[5,6], [6,7],[7,4],
#             [8,9],[9,10],[10,11],[11,8],
#             [12,13],[13,14],[14,15],[15,12],
#             [16,17],[17,18],[18,19],[19,16]]
v = [[0, 0], [1, 0], [1, 1], [0, 1]]
segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
para = "pq30a0.01e"
t = triangulate({"vertices": v, 'segments': segments}, para)
Mesh = t["triangles"]
points = t["vertices"]
edges = t["edges"]
Nv = points.shape[0]
Nelt = Mesh.shape[0]

######################################################################################
# create list of vertices, element and face
meshelt = []
for i in range(Nelt):
    tmpelt = element(Mesh[i])
    meshelt.append(tmpelt)

meshvertex = []
for i in range(Nv):
    meshvertex.append(points[i])

meshface = []
Nface = edges.shape[0]
Nif = 0; Nbf = 0; Nibf = 0; Nfbf = 0
for i in range(Nface):
    neighbor = [] # two cells alone the edge
    for k in range(Nelt):
        # assemble adjacency matrix and edge info
        if (edges[i][0] in Mesh[k]) and (edges[i][1] in Mesh[k]):
            neighbor.append(k)
    if len(neighbor) == 2:
        e = face(edges[i], neighbor)
        meshface.append(e)
        Nif += 1
    elif len(neighbor) == 1:
        e = face(edges[i], neighbor)
        if meshvertex[edges[i][0]][1]<1e-9 and meshvertex[edges[i][1]][1]<1e-9:
            e.bctype = -1
            Nibf += 1
        elif abs(meshvertex[edges[i][0]][1] - 1)<1e-9 and abs(1 - meshvertex[edges[i][1]][1])<1e-9:
            e.bctype = -2
            Nfbf += 1
        else:
            e.bctype = 1
            Nbf += 1
        meshface.append(e)

######################################################################################
# print information about the mesh
print(Nv, 'points in the whole domain')
print(Nface, 'faces in the whole domain')
print(Nelt, 'elements in the whole domain')
print(Nif, 'interior edges;', Nbf, 'dirichlet boundary edges;', Nibf, 'inital t boundary;', Nfbf, 'final t boundary;')

print('Mesh:',Mesh)
print('points (x, t):', points)
print('edges:', edges)


######################################################################################
# plot the mesh
print("Plot the mesh")
plt.triplot(points[:,0], points[:,1], Mesh) 
plt.plot(points[:,0], points[:,1], 'o') 
plt.axis('equal')
plt.show() 

phi = {0:phi0, 1:phi1, 2:phi2, 3:phi3, 4:phi4, 5:phi5}
grad_phi = {0:grad_phi0, 1:grad_phi1, 2:grad_phi2, 3:grad_phi3, 4:grad_phi4, 5:grad_phi5}

if __name__ == "__main__":
    p = np.array([0.5, 0.5])
    print(source(p))