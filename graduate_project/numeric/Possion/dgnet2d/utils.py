import numpy as np
import torch
from torch import tensor as Tensor
import matplotlib.pyplot as plt
from mesh import *
from math import exp
###############################################################################################
def is_in_ref(p):
    '''
    whether a point is in the reference cell
    in reference triangle: 1
    on the boundary: 0
    outside: -1
    '''
    if 1-(p[0]+p[1])>1e-9 and p[0]>-1e-10 and p[1]>-1e-10:
        return 1
    elif abs(p[0]+p[1]-1) < 1e-9 and p[0]>-1e-9 and p[1]>-1e-9:
        return 0
    elif abs(p[0])<1e-9 and p[1]>-1e-9 and abs(1-p[1])>1e-9:
        return 0
    elif abs(p[1])<1e-9 and p[0]>-1e-9 and abs(1-p[0])>1e-9:
        return 0
    else:
        print('error')
        return -1
    
def computeBE(p1, p2, p3):
    '''compute B_E:hat(E) to E, accept three 1*2 np array, return 2*2 matrix'''
    BE = Tensor([[p2[0]-p1[0], p3[0]-p1[0]], [p2[1]-p1[1], p3[1]-p1[1]]])
    bE = p1.clone()
    return BE, bE


def hat2E(BE, bE, p): 
    return torch.matmul(BE, p) + bE  
  
def E2hat(BE, bE, p):  
    invB = torch.inverse(BE)  
    refcor = torch.matmul(invB, p - bE)  
    return refcor  
  
def length(p1, p2):  
    return torch.norm(p2 - p1)  
  
def I2ref(p1, p2, p):  
    v = p2 - p1  
    if torch.equal(p, p1):  
        return Tensor(-1.0)  
    elif torch.equal(p, p2):  
        return Tensor(1.0)  
    else:  
        v0 = p - p1  
        re = v0 / v  
        return re[0] * 2.0 - 1.0  
  
def ref2I(p1, p2, x):  
    v = p2 - p1  
    return (x + 1.0) / 2.0 * v + p1 

def sample_triangle(n):
    """在直角三角形上均匀采样点"""
    # 生成参数化的直角三角形上的均匀采样点
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    mask = 1 - X - Y > -1e-9
    points = np.column_stack((X[mask], Y[mask]))
    return points

def sample_edge(n):
    x = np.linspace(-1, 1, n)
    return x

def plot_true_3D():
    x = np.linspace(0, 1, 100)  
    y = np.linspace(0, 1, 100)  
    X, Y = np.meshgrid(x, y)  
    
    # 计算函数的值  
    Z = np.exp(-X**2 - Y**2)  
    
    # 绘制图像  
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')  
    ax.plot_surface(X, Y, Z, cmap='jet')  
    
    # 设置坐标轴标签和标题  
    ax.set_xlabel('x')  
    ax.set_ylabel('y')  
    ax.set_zlabel('e^(-x^2-y^2)')  
    ax.set_title('Graph of e^(-x^2-y^2)')  
    filename = f'./pic/true3D.png'  
    plt.savefig ( filename )
    print ( '' )
    print ( '  Graphics saved as "%s"' % ( filename ) )
    # 显示图像  
    plt.show() 

def plot_true_heat():
    x = np.linspace(0, 1, 100)  
    y = np.linspace(0, 1, 100)  
    X, Y = np.meshgrid(x, y)  
    
    # 计算函数的值  
    Z = np.exp(-X**2 - Y**2)  
    
    # 绘制热力图  
    plt.figure()  
    plt.contourf(X, Y, Z,cmap='jet')  
    plt.colorbar(label='e^(-x^2-y^2)')  
    plt.xlabel('x')  
    plt.ylabel('y')  
    plt.title('Heatmap of e^(-x^2-y^2)')  
    filename = f'./pic/trueheat.png'  
    plt.savefig ( filename )
    print ( '' )
    print ( '  Graphics saved as "%s"' % ( filename ) )
    # 显示图像  
    plt.show()  

def plot_basis():
    x = np.linspace(0, 1, 100)  
    y = (1-x)**6/3*(35*x**2+18*x+3)
    # y = np.exp(-(x**2)/2.0)
    plt.figure()  
    plt.plot(x, y, color='b')  
    plt.xlabel('x')  
    plt.ylabel('y')  
    plt.title('Heatmap of e^(-x^2-y^2)')  
    plt.show() 


if __name__ == "__main__":
    n = 5
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    mask = 1 - X - Y > -1e-9
    points = np.column_stack((X[mask], Y[mask]))

    # 绘制符合条件的点
    plt.scatter(points[:, 0], points[:, 1], cmap='jet')

    # 设置图形标题和坐标轴标签
    plt.title('Points in 2D Plane')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.grid(True)
    plt.show()
    plot_basis()

