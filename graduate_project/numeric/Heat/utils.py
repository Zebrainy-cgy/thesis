import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
from mesh import *
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
        print('outside: ?')
        return -1

def computeBE(p1, p2, p3):
    '''compute B_E:hat(E) to E, accept three 1*2 np array, return 2*2 matrix'''
    BE = np.array([[p2[0]-p1[0], p3[0]-p1[0]], [p2[1]-p1[1], p3[1]-p1[1]]])
    bE = p1
    return BE, bE

def hat2E(BE, bE, p):
    return np.dot(BE, p) + bE

def E2hat(BE, bE, p):
    invB = np.linalg.inv(BE)
    refcor = np.dot(invB, p - bE)
    return refcor

def length(p1, p2):
    return np.linalg.norm(p2 - p1)


def I2ref(p1, p2, p):
    '''map points on p1<->p2 to [-1, 1]'''
    v = p2 - p1
    if np.array_equal(p, p1):
        # print('p12-1',p1)
        return -1
    elif np.array_equal(p, p2):
        # print('p221',p2)
        return 1
    else:
        v0 = p - p1
        re = v0/v
        # print(v, v0, re)
        return re[0] * 2.0 - 1

# p2 = np.array([1,0])
# print(normvec(p1, p2))
def ref2I(p1, p2, x):
    '''map points on reference interval [-1,1] to points on edge'''
    v = p2 - p1
    return (x + 1.0)/2.0 * v + p1

def sample_triangle(n):
    """在直角三角形上均匀采样点"""
    # 生成参数化的直角三角形上的均匀采样点
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    mask = 1 - X - Y > -1e-9
    points = np.column_stack((X[mask], Y[mask]))
    return points

def plot_true_3D():
    x = np.linspace(0, 1, 100)  
    t = np.linspace(0, 1, 100)  
    X, T = np.meshgrid(x, t)  
    
    # 计算函数的值  
    # Z = np.cos(pi*T)*np.sin(pi*X) 
    Z = (X**2+(T-1.0/2)**2)**(1/4)
    
    # 绘制图像  
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d') 
    ax.view_init(elev=40, azim=220) 
    ax.plot_surface(X, T, Z, cmap='jet')  
    
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
    t = np.linspace(0, 1, 100)  
    X, T = np.meshgrid(x, t)  
    
    # 计算函数的值  
    # Z = np.cos(pi*T)*np.sin(pi*X) 
    Z = (X**2+(T-1.0/2)**2)**(1/4)
    
    # 绘制热力图  
    plt.figure()  
    plt.contourf(X, T, Z,cmap='jet')  
    plt.colorbar(label='(X**2+(T-1.0/2)**2)**(1/4)')  
    plt.xlabel('x')  
    plt.ylabel('y')  
    plt.title('Heatmap of (X**2+(T-1.0/2)**2)**(1/4)')  
    filename = f'./pic/trueheat.png'  
    plt.savefig ( filename )
    print ( '' )
    print ( '  Graphics saved as "%s"' % ( filename ) )
    # 显示图像  
    plt.show()  

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 5
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    mask = 1 - X - Y > -1e-9
    points = np.column_stack((X[mask], Y[mask]))

    # 绘制符合条件的点
    plt.scatter(points[:, 0], points[:, 1], color='red', marker='.')

    # 设置图形标题和坐标轴标签
    plt.title('Points in 2D Plane')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.grid(True)
    plt.show()

