import numpy as np
import matplotlib.pyplot as plt
from math import exp
from matplotlib.tri import Triangulation
from triangle import triangulate
from mesh import *
from bf import *
deg = 2
Nloc = (deg + 1) * (deg + 2) // 2 # 单元上函数空间维数

print('local dimention:', Nloc)
eps = -1.0 # determine SIPG or NIPG
sigma = 10.0 * deg * deg # penalty variant

# set condition
##########################################################################
#homo
# def source(realcor):
#     return 10

# def dirichlet(realcor):
#     return 0
##########################################################################
# smooth
# def source(realcor):
#     x = realcor[0]; y = realcor[1]
#     return 4 * (1-x**2-y**2) * exp(-x**2-y**2)

# def dirichlet(realcor):
#     x = realcor[0]; y = realcor[1]
#     return exp(-x**2-y**2)

##########################################################################
# jump sol
def source(realcor):
    x = realcor[0]+1e-4; y = realcor[1]+1e-4
    return -0.25*(x**2+(y-0.5)**2)**(-3/4)

def dirichlet(realcor):
    x = realcor[0]+1e-4; y = realcor[1]+1e-4
    return (x**2+(y-0.5)**2)**(1/4)


##########################################################################
# square doamin
v = [[0, 0], [0, 1], [1, 1], [1, 0]]
segments = [[0, 1], [1, 2], [2, 3], [3, 0]]

# irregular domain
# v = [[0, 0], [0, 2], [1, 2], [1, 1],[2,1],[2,0]]
# segments = [[0, 1], [1, 2], [2, 3], [3, 4],[4,5],[5,0]]

para = "pq30a0.01e"
t = triangulate({"vertices": v, 'segments': segments,}, para)
Mesh = t["triangles"]
points = t["vertices"]
edges = t["edges"]
Nv = points.shape[0]
Nelt = Mesh.shape[0]


meshelt = []
for i in range(Nelt):
    tmpelt = element(Mesh[i])
    meshelt.append(tmpelt)
    # print(meshelt[i].vertex)

meshvertex = []
for i in range(Nv):
    meshvertex.append(points[i])

meshface = []

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
        Nif += 1
        # print('interior edge', i, j, 'is edge of cell', neighbor)
    elif len(neighbor) == 1:
        e = face(edges[i], neighbor)
        e.bctype = 1
        Nbf += 1
        meshface.append(e)


print(Nv, 'points in the whole domain')
print(Nface, 'faces in the whole domain')
print(Nelt, 'elements in the whole domain')
print(Nif, 'interior edges', Nbf, 'boundary edges')

print("Plot the mesh")
plt.triplot(points[:,0], points[:,1], Mesh) 
plt.plot(points[:,0], points[:,1], 'o') 
plt.axis('equal')
plt.show() 

phii = {0:phi0, 1:phi1, 2:phi2, 3:phi3, 4:phi4, 5:phi5}
grad_phii = {0:grad_phi0, 1:grad_phi1, 2:grad_phi2, 3:grad_phi3, 4:grad_phi4, 5:grad_phi5}
# gbfi = grad_phii.get(4, "Invalid")(np.array([0.99999999,0]))

if __name__ == "__main__":
# plot mesh
    print('Mesh:',Mesh)
    print('points:', points)
    print('edges:', edges, Nface)
    p = np.array([0.5, 0.5])
    print(source(p))
    plt.triplot(points[:,0], points[:,1], Mesh) 
    plt.plot(points[:,0], points[:,1], 'o') 
    plt.show() 