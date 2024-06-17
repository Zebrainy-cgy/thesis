import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh import element, face
from bf import *
from condition import *

################################################################################################################
def basisfunction(i, E: element, p):
    '''accept real coordinates and return phi_i^E(x) x \in E, defined by hat phi'''
    p1 = meshvertex[E.vertex[0]]; p2 = meshvertex[E.vertex[1]]; p3 = meshvertex[E.vertex[2]]
    BE, bE = computeBE(p1, p2, p3)
    refcor = E2hat(BE, bE, p)
    if is_in_ref(refcor) != -1:
        phi = phii.get(i, "Invalid")
        return phi(refcor)
    else:
        return 0

def grad_basisfunction(i, E: element, p):
    p1 = meshvertex[E.vertex[0]]; p2 = meshvertex[E.vertex[1]]; p3 = meshvertex[E.vertex[2]]
    BE, bE = computeBE(p1, p2, p3)
    refcor = E2hat(BE, bE, p)
    if is_in_ref(refcor) != -1:
        grad_phi = grad_phii.get(i, "Invalid")
        result = np.dot(np.linalg.inv(BE).T, grad_phi)
        return result
    else:
        return np.zeros(2)
    
################################################################################################################
def computeAEFE(E: element):
    '''A_E and F_E, neglect alpha phi_i*phi_j'''
    AE = np.zeros((Nloc, Nloc))
    FE = np.zeros(Nloc)
    w = np.array([1.0/6, 1.0/6, 1.0/6])
    points = np.array([[2.0/3, 1.0/6], [1.0/6, 1.0/6], [1.0/6, 2.0/3]])
        # print(w)
    NG = w.shape[0] # 积分估计点数

    p1 = meshvertex[E.vertex[0]]; p2 = meshvertex[E.vertex[1]]; p3 = meshvertex[E.vertex[2]]
    BE, bE = computeBE(p1, p2, p3)
    invBE = np.linalg.inv(BE).T
    
    for k in range(NG):
        detBE = np.linalg.det(BE)   
        realcor = hat2E(BE, bE, points[k])
        f = source(realcor)
        for i in range(Nloc):
            bfi = phii.get(i, "Invalid")(points[k])
            gbfi = np.dot(invBE, grad_phii.get(i, "Invalid")(points[k]))
            for j in range(Nloc):
                gbfj = np.dot(invBE, grad_phii.get(j, "Invalid")(points[k]))
                AE[i][j] += w[k] * detBE * np.dot(gbfi, gbfj)
            FE[i] += w[k] * detBE * f * bfi
    return AE, FE

################################################################################################################
def normvec(e: face):
    '''norm vector from E1 point to E2'''
    p1 = meshvertex[e.vertex[0]]; p2 = meshvertex[e.vertex[1]]
    v0 = p2 - p1
    if e.bctype == 0:
        E1 = meshelt[e.neighbor[0]]; E2 = meshelt[e.neighbor[1]]
        # print(e.neighbor[0], e.neighbor[1])
        inx = np.array([e.vertex[0], e.vertex[1]])
        n1 = E1.vertex[~np.isin(E1.vertex, inx)][0]
        n2 = E2.vertex[~np.isin(E2.vertex, inx)][0]
        # print(n1, n2)
        v1 = meshvertex[n2] - meshvertex[n1]
        tmpvec = np.array([-v0[1], v0[0]])
        t = np.dot(v1, tmpvec)
        if t > -1e-10:
            return tmpvec / np.linalg.norm(tmpvec)
        else:
            return -tmpvec / np.linalg.norm(tmpvec)
    elif e.bctype == 1:
        E1 = meshelt[e.neighbor[0]]
        inx = np.array([e.vertex[0], e.vertex[1]])
        n1 = E1.vertex[~np.isin(E1.vertex, inx)][0]
        v1 = meshvertex[n1] - p1
        tmpvec = np.array([-v0[1], v0[0]])
        t = np.dot(v1, tmpvec)
        if t > -1e-10:
            return -tmpvec / np.linalg.norm(tmpvec)
        else:
            return tmpvec / np.linalg.norm(tmpvec)
        

################################################################################################################
def computeM(e: face):
    '''local M, return M11, M22, M12, M21, give two elements alone the edge'''
    p1 = meshvertex[e.vertex[0]]; p2 = meshvertex[e.vertex[1]]
    leng = length(p1, p2)
    nvec = normvec(e)
    s = np.array([-0.86113631, -0.33998104, 0.33998104, 0.86113631])  
    wE = np.array([0.34785485, 0.65214515, 0.65214515, 0.34785485]) / 2
    NGE = wE.shape[0]
    
    if e.bctype == 0:
        # e is interior edge
        M11 = np.zeros((Nloc, Nloc))
        M22 = np.zeros((Nloc, Nloc))
        M21 = np.zeros((Nloc, Nloc))
        M12 = np.zeros((Nloc, Nloc))
        E1 = meshelt[e.neighbor[0]]; E2 = meshelt[e.neighbor[1]]
        BE1, bE1 = computeBE(meshvertex[E1.vertex[0]], meshvertex[E1.vertex[1]], meshvertex[E1.vertex[2]])
        BE2, bE2 = computeBE(meshvertex[E2.vertex[0]], meshvertex[E2.vertex[1]], meshvertex[E2.vertex[2]])
        invBE1 = np.linalg.inv(BE1).T
        invBE2 = np.linalg.inv(BE2).T

        for k in range(NGE):
            realcor = ref2I(p1, p2, s[k])
            refcorE1 = E2hat(BE1, bE1, realcor)
            refcorE2 = E2hat(BE2, bE2, realcor)
            # compute M_k^{11}
            for i in range(Nloc):
                bfi1 = phii.get(i, "Invalid")(refcorE1)
                if grad_phii.get(i, "Invalid")(refcorE1) is None:
                    print(k, realcor, refcorE1, grad_phii.get(i, "Invalid")(refcorE1))
                gbfi1 = np.dot(invBE1, grad_phii.get(i, "Invalid")(refcorE1))
                for j  in range(Nloc):
                    bfj1 = phii.get(j, "Invalid")(refcorE1)
                    gbfj1 = np.dot(invBE1, grad_phii.get(j, "Invalid")(refcorE1))
                    M11[i][j] += -0.5 * wE[k] * leng * bfi1 * np.dot(gbfj1, nvec)
                    M11[i][j] += 0.5 * eps * wE[k] * leng * bfj1 * np.dot(gbfi1, nvec)
                    M11[i][j] += sigma * wE[k] * bfi1 * bfj1
            # compute M_k^{22}
            for i in range(Nloc):
                bfi2 = phii.get(i, "Invalid")(refcorE2)
                gbfi2 = np.dot(invBE2, grad_phii.get(i, "Invalid")(refcorE2))
                for j  in range(Nloc):
                    bfj2 = phii.get(j, "Invalid")(refcorE2)
                    gbfj2 = np.dot(invBE2, grad_phii.get(j, "Invalid")(refcorE2))
                    M22[i][j] += 0.5 * wE[k] * leng * bfi2 * np.dot(gbfj2, nvec)
                    M22[i][j] += -0.5 * eps * wE[k] * leng * bfj2 * np.dot(gbfi2, nvec)
                    M22[i][j] += sigma * wE[k] * bfi2 * bfj2
            # compute M_k^{12}
            for i in range(Nloc):
                bfi1 = phii.get(i, "Invalid")(refcorE1)
                gbfi1 = np.dot(invBE1, grad_phii.get(i, "Invalid")(refcorE1))
                for j  in range(Nloc):
                    bfj2 = phii.get(j, "Invalid")(refcorE2)
                    gbfj2 = np.dot(invBE2, grad_phii.get(j, "Invalid")(refcorE2))
                    M12[i][j] += -0.5 * wE[k] * leng * bfi1 * np.dot(gbfj2, nvec)
                    M12[i][j] += -0.5 * eps * wE[k] * leng * bfj2 * np.dot(gbfi1, nvec)
                    M12[i][j] += -sigma * wE[k] * bfi1 * bfj2
            # compute M_k^{21}
            for i in range(Nloc):
                bfi2 = phii.get(i, "Invalid")(refcorE2)
                gbfi2 = np.dot(invBE2, grad_phii.get(i, "Invalid")(refcorE2))
                for j  in range(Nloc):
                    bfj1 = phii.get(j, "Invalid")(refcorE1)
                    gbfj1 = np.dot(invBE1, grad_phii.get(j, "Invalid")(refcorE1))
                    M21[i][j] += 0.5 * wE[k] * leng * bfi2 * np.dot(gbfj1, nvec)
                    M21[i][j] += 0.5 * eps * wE[k] * leng * bfj1 * np.dot(gbfi2, nvec)
                    M21[i][j] += -sigma * wE[k] * bfi2 * bfj1
        # print(length(np.array([1, 0]), np.array([0, 1])), normvec(np.array([1, 0]), np.array([0, 1])))
        # return M11, M22, M12, M21
        return M11, M22, M12, M21
    elif e.bctype == 1:
        # e is boundary edge
        M11 = np.zeros((Nloc, Nloc))
        be = np.zeros(Nloc)
        E1 = meshelt[e.neighbor[0]]
        BE1, bE1 = computeBE(meshvertex[E1.vertex[0]], meshvertex[E1.vertex[1]], meshvertex[E1.vertex[2]])
        invBE1 = np.linalg.inv(BE1).T

        for k in range(NGE):
            # compute M_k^{11}
            realcor = ref2I(p1, p2, s[k])
            refcorE1 = E2hat(BE1, bE1, realcor)
            for i in range(Nloc):
                bfi1 = phii.get(i, "Invalid")(refcorE1) 
                gbfi1 = np.dot(invBE1, grad_phii.get(i, "Invalid")(refcorE1))
                for j  in range(Nloc):
                    bfj1 = phii.get(j, "Invalid")(refcorE1)
                    gbfj1 = np.dot(invBE1, grad_phii.get(j, "Invalid")(refcorE1))
                    M11[i][j] += - wE[k] * leng * bfi1 * np.dot(gbfj1, nvec)
                    M11[i][j] += eps * wE[k] * leng * bfj1 * np.dot(gbfi1, nvec)
                    M11[i][j] += sigma * wE[k] * bfi1 * bfj1
                be[i] += wE[k] * (eps * leng * np.dot(gbfi1, nvec) + sigma * bfi1) * dirichlet(realcor)
        # return M11, be
        return M11, be
    

################################################################################################################
Ntol = Nelt * Nloc
Fglobal = np.zeros(Ntol)
Aglobal = np.zeros((Ntol, Ntol))
A = np.zeros((Ntol, Ntol))
for k in range(Nelt):
    E = meshelt[k]
    AE, FE = computeAEFE(E)
    # print(AE)
    # print(FE)
    for i in range(Nloc):
        ie = i + k * Nloc
        # print(i, k)
        Fglobal[ie] += FE[i]
        for j in range(Nloc):
            je = j + k * Nloc
            Aglobal[ie][je] += AE[i][j]
            
            # print(ie, je, i, j)
# print(Fglobal.reshape(Nelt, Nloc))


for k in range(Nface):
    e = meshface[k]
    if meshface[k].bctype == 0:
        noe1 = e.neighbor[0]
        noe2 = e.neighbor[1]
        # print(noe1, noe2)
        M11, M22, M12, M21 = computeM(e)
        for i in range(Nloc):
            ie = i + noe1 * Nloc
            for j in range(Nloc):
                je = j + noe1 * Nloc
                Aglobal[ie][je] += M11[i][j]

                
        for i in range(Nloc):
            ie = i + noe2 * Nloc
            for j in range(Nloc):
                je = j + noe2 * Nloc
                Aglobal[ie][je] += M22[i][j]
                
        for i in range(Nloc):
            ie = i + noe1 * Nloc
            for j in range(Nloc):
                je = j + noe2 * Nloc
                Aglobal[ie][je] += M12[i][j]
                
        for i in range(Nloc):
            ie = i + noe2 * Nloc
            for j in range(Nloc):
                je = j + noe1 * Nloc
                Aglobal[ie][je] += M21[i][j]
                
    elif meshface[k].bctype == 1:
        noe1 = e.neighbor[0]
        M11, be = computeM(e)
        for i in range(Nloc):
            ie = i + noe1 * Nloc
            Fglobal[ie] += be[i]
            for j in  range(Nloc):
                je = j + noe1 * Nloc
                Aglobal[ie][je] += M11[i][j]
                

u = np.linalg.solve(Aglobal, Fglobal)
u = u.reshape(Nelt, Nloc)
print("solution:", u)
print("solution shape:", u.shape)

def solution(coe, k, p):
    '''p is real coordinates'''
    sol = 0
    E = meshelt[k]
    for i in range(Nloc):
        sol += basisfunction(i, E, p) * coe[i]
        # sol += basisfunction(i, E, p)
    return sol


def plot3D(): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.view_init(elev=40, azim=220)
    n_inter = 5
    p_inter = sample_triangle(n_inter)
    N_inter = p_inter.shape[0]
    N_tol = Nelt * N_inter
    sol = np.zeros(N_tol)
    plot_p = np.zeros((N_tol, 2))
    for i in range(Nelt):
        for j in range(N_inter):
            n1 = Mesh[i][0]; n2 = Mesh[i][1]; n3 = Mesh[i][2]
            p1 = meshvertex[n1]; p2 = meshvertex[n2]; p3 = meshvertex[n3]; 
            BE, bE = computeBE(p1, p2, p3)
            ie = i * N_inter + j
            real_p = hat2E(BE, bE, p_inter[j])
            sol[ie] = solution(u[i], i, real_p)
            plot_p[ie] = real_p

    ax.plot_trisurf(plot_p[:, 0], plot_p[:, 1], sol, cmap='jet')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Possion Equation')
    # ax.grid(False)

    # vertices = np.array([[1, 1, 0],
    #                      [1, 2, 0],
    #                      [2,2,0],
    #                      [2, 1, 0]])

    # # 定义三角形的面
    # faces = [[vertices[j] for j in [0, 1, 2, 3]]]

    # # 绘制三角形
    # triangle = Poly3DCollection(faces, facecolors='white', alpha=1.0)
    # ax.add_collection3d(triangle)

    filename = f'./pic/3Dplot_eps{eps}sig{sigma}_para{para}.png'  
    plt.savefig ( filename )
    print ( '' )
    print ( '  Graphics saved as "%s"' % ( filename ) )
    plt.show()


def plot_heatmap():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_inter = 5
    p_inter = sample_triangle(n_inter)
    N_inter = p_inter.shape[0]
    N_tol = Nelt * N_inter
    sol = np.zeros(N_tol)
    plot_p = np.zeros((N_tol, 2))
    for i in range(Nelt):
        for j in range(N_inter):
            n1 = Mesh[i][0]; n2 = Mesh[i][1]; n3 = Mesh[i][2]
            p1 = meshvertex[n1]; p2 = meshvertex[n2]; p3 = meshvertex[n3]; 
            BE, bE = computeBE(p1, p2, p3)
            ie = i * N_inter + j
            real_p = hat2E(BE, bE, p_inter[j])
            sol[ie] = solution(u[i], i, real_p)
            plot_p[ie] = real_p
    
    ax.tricontourf(plot_p[:, 0], plot_p[:, 1], sol, cmap='jet')
    # ax.fill([1, 1, 2, 2], [1, 2, 2, 1], color='white')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Possion Equation Heatmap (2D)')
    plt.axis('equal')

    filename = f'./pic/heatmap_eps{eps}sig{sigma}_para{para}.png'  
    plt.savefig ( filename )
    print ( '' )
    print ( '  Graphics saved as "%s"' % ( filename ) )
    plt.show()




plot3D()
plot_heatmap()
# plot_true_3D()
# plot_true_heat()
