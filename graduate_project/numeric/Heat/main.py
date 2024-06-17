import numpy as np
import matplotlib.pyplot as plt
from mesh import *
from bf import *
from condition import *
from utils import *

def computeAF(E: element):
    A = np.zeros((Nloc, Nloc))
    F = np.zeros(Nloc)
    w = np.array([1.0/6, 1.0/6, 1.0/6])
    points = np.array([[2.0/3, 1.0/6], [1.0/6, 1.0/6], [1.0/6, 2.0/3]])
    NG = w.shape[0]
    p1 = meshvertex[E.vertex[0]]; p2 = meshvertex[E.vertex[1]]; p3 = meshvertex[E.vertex[2]]
    BE, bE = computeBE(p1, p2, p3)
    invBE = np.linalg.inv(BE)
    detBE = np.linalg.det(BE)   
    for k in range(NG):
        realcor = hat2E(BE, bE, points[k])
        f = source(realcor)
        for i in range(Nloc):
            phii = phi.get(i)(points[k])
            phiix = np.dot(invBE[:, 0], grad_phi.get(i)(points[k]))
            phiit = np.dot(invBE[:, 1], grad_phi.get(i)(points[k]))
            for j in range(Nloc):
                phijx = np.dot(invBE[:, 0], grad_phi.get(j)(points[k]))
                phij = phi.get(j)(points[k])
                A[i][j] += detBE * w[k] * phiix * phijx
                A[i][j] -= detBE * w[k] * phij * phiit
            F[i] += w[k] * detBE * f * phii
    return A, F

def normvec(e: face):
    '''
    if e is interior edge, then norm vector point from E1 point to E2
    if e is boundary edge, then norm vector point out E1
    nx is normvec[0], nt is normvec[1]
    '''
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
    elif e.bctype != 0:
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

def computeM(e:face):
    p1 = meshvertex[e.vertex[0]]; p2 = meshvertex[e.vertex[1]]
    leng = length(p1, p2)
    nvec = normvec(e)
    s = np.array([-0.86113631, -0.33998104, 0.33998104, 0.86113631])  
    w = np.array([0.34785485, 0.65214515, 0.65214515, 0.34785485]) / 2
    NG = w.shape[0]
    if e.bctype == 0:
        # e is interior edge
        M11 = np.zeros((Nloc, Nloc))
        M22 = np.zeros((Nloc, Nloc))
        M21 = np.zeros((Nloc, Nloc))
        M12 = np.zeros((Nloc, Nloc))
        E1 = meshelt[e.neighbor[0]]; E2 = meshelt[e.neighbor[1]]
        BE1, bE1 = computeBE(meshvertex[E1.vertex[0]], meshvertex[E1.vertex[1]], meshvertex[E1.vertex[2]])
        BE2, bE2 = computeBE(meshvertex[E2.vertex[0]], meshvertex[E2.vertex[1]], meshvertex[E2.vertex[2]])
        invBE1 = np.linalg.inv(BE1)
        invBE2 = np.linalg.inv(BE2)
        for k in range(NG):
            realcor = ref2I(p1, p2, s[k])
            refcorE1 = E2hat(BE1, bE1, realcor)
            refcorE2 = E2hat(BE2, bE2, realcor)
            # compute M_k^{11}
            for i in range(Nloc):
                phii1 = phi.get(i)(refcorE1)
                phii1x = np.dot(invBE1[:, 0], grad_phi.get(i)(refcorE1))
                for j in range(Nloc):
                    phij1 = phi.get(j)(refcorE1)
                    phij1x = np.dot(invBE1[:, 0], grad_phi.get(j)(refcorE1))
                    M11[i][j] += -0.5 * w[k] * leng * nvec[0] * phij1x * phii1
                    M11[i][j] += eps * 0.5 * w[k] * leng * phii1x * phij1 * nvec[0]
                    M11[i][j] += sigma * w[k] * phii1 * nvec[0] * phij1 * nvec[0]
            # compute M_k^{22}
            for i in range(Nloc):
                phii2 = phi.get(i)(refcorE2)
                phii2x = np.dot(invBE2[:, 0], grad_phi.get(i)(refcorE2))
                for j in range(Nloc):
                    phij2 = phi.get(j)(refcorE2)
                    phij2x = np.dot(invBE2[:, 0], grad_phi.get(j)(refcorE2))
                    M22[i][j] += 0.5 * w[k] * leng * nvec[0] * phij2x * phii2
                    M22[i][j] += -eps * 0.5 * w[k] * leng * phii2x * phij2 * nvec[0]
                    M22[i][j] += sigma * w[k] * phii2 * nvec[0] * phij2 * nvec[0]
            # compute M_k^{12}
            for i in range(Nloc):
                phii1 = phi.get(i)(refcorE1)
                phii1x = np.dot(invBE1[:, 0], grad_phi.get(i)(refcorE1))
                for j in range(Nloc):
                    phij2 = phi.get(j)(refcorE2)
                    phij2x = np.dot(invBE2[:, 0], grad_phi.get(j)(refcorE2))
                    M12[i][j] += -0.5 * w[k] * leng * nvec[0] * phij2x * phii1
                    M12[i][j] += -eps * 0.5 * w[k] * leng * phii1x * phij2 * nvec[0]
                    M12[i][j] += -sigma * w[k] * phii1 * nvec[0] * phij2 * nvec[0]
            # compute M_k^{21}
            for i in range(Nloc):
                phii2 = phi.get(i)(refcorE2)
                phii2x = np.dot(invBE2[:, 0], grad_phi.get(i)(refcorE2))
                for j in range(Nloc):
                    phij1 = phi.get(j)(refcorE1)
                    phij1x = np.dot(invBE1[:, 0], grad_phi.get(j)(refcorE1))
                    M21[i][j] += 0.5 * w[k] * leng * nvec[0] * phij1x * phii2
                    M21[i][j] += eps * 0.5 * w[k] * leng * phii2x * phij1 * nvec[0]
                    M21[i][j] += -sigma * w[k] * phii2 * nvec[0] * phij1 * nvec[0]
        return M11, M22, M12, M21
    
    elif e.bctype == 1:
        # e is dirchlet boundary edge
        M11 = np.zeros((Nloc, Nloc))
        be = np.zeros(Nloc)
        E = meshelt[e.neighbor[0]]
        BE, bE = computeBE(meshvertex[E.vertex[0]], meshvertex[E.vertex[1]], meshvertex[E.vertex[2]])
        invBE = np.linalg.inv(BE)

        for k in range(NG):
            realcor = ref2I(p1, p2, s[k])
            refcorE = E2hat(BE, bE, realcor)
            # compute M_k^{11}
            for i in range(Nloc):
                phii = phi.get(i)(refcorE)
                phiix = np.dot(invBE[:, 0], grad_phi.get(i)(refcorE))
                for j in range(Nloc):
                    phij = phi.get(j)(refcorE)
                    phijx = np.dot(invBE[:, 0], grad_phi.get(j)(refcorE))
                    M11[i][j] += -w[k] * leng * nvec[0] * phijx * phii
                    M11[i][j] += eps * w[k] * leng * phiix * phij * nvec[0]
                    M11[i][j] += sigma * w[k] * phii * nvec[0] * phij * nvec[0]
                be[i] += w[k] * (eps * leng * phiix * nvec[0] + sigma * phii * nvec[0] * nvec[0]) * dirichlet(refcorE)
        # return M11, be
        return M11, be
    
def computeN(e: face):
    p1 = meshvertex[e.vertex[0]]; p2 = meshvertex[e.vertex[1]]
    leng = length(p1, p2)
    nvec = normvec(e)
    nt = nvec[1]
    s = np.array([-0.86113631, -0.33998104, 0.33998104, 0.86113631])  
    w = np.array([0.34785485, 0.65214515, 0.65214515, 0.34785485]) / 2
    NG = w.shape[0]
    if e.bctype == 0:
        E1 = meshelt[e.neighbor[0]]; E2 = meshelt[e.neighbor[1]]
        BE1, bE1 = computeBE(meshvertex[E1.vertex[0]], meshvertex[E1.vertex[1]], meshvertex[E1.vertex[2]])
        BE2, bE2 = computeBE(meshvertex[E2.vertex[0]], meshvertex[E2.vertex[1]], meshvertex[E2.vertex[2]])
        invBE1 = np.linalg.inv(BE1)
        invBE2 = np.linalg.inv(BE2)
        if nt > 1e-10:
            # print(nt)
            N11 = np.zeros((Nloc, Nloc))
            N21 = np.zeros((Nloc, Nloc))
            for k in range(NG):
                realcor = ref2I(p1, p2, s[k])
                refcorE1 = E2hat(BE1, bE1, realcor)
                refcorE2 = E2hat(BE2, bE2, realcor)
                for i in range(Nloc):
                    phii1 = phi.get(i)(refcorE1)
                    phii2 = phi.get(i)(refcorE2)
                    for j in range(Nloc):
                        phij1 = phi.get(j)(refcorE1)
                        N11[i][j] += w[k] * leng * phii1 * phij1 * nt
                        N21[i][j] -= w[k] * leng * phii2 * phij1 * nt
            return N11, N21
        elif abs(nt) < 1e-10:
            pass
        elif nt < -1e-10:
            N22 = np.zeros((Nloc, Nloc))
            N12 = np.zeros((Nloc, Nloc))
            for k in range(NG):
                realcor = ref2I(p1, p2, s[k])
                refcorE1 = E2hat(BE1, bE1, realcor)
                refcorE2 = E2hat(BE2, bE2, realcor)
                for i in range(Nloc):
                    phii1 = phi.get(i)(refcorE1)
                    phii2 = phi.get(i)(refcorE2)
                    for j in range(Nloc):
                        phij2 = phi.get(j)(refcorE2)
                        N12[i][j] += w[k] * leng * phii1 * phij2 * nt
                        N22[i][j] -= w[k] * leng * phii2 * phij2 * nt
            return N12, N22
    elif e.bctype == -1:
        # sigma_0 u0
        be = np.zeros(Nloc)
        E = meshelt[e.neighbor[0]]
        BE, bE = computeBE(meshvertex[E.vertex[0]], meshvertex[E.vertex[1]], meshvertex[E.vertex[2]])
        for k in range(NG):
            realcor = ref2I(p1, p2, s[k])
            refcorE = E2hat(BE, bE, realcor)
            iu = initu(realcor)
            for i in range(Nloc):
                phii = phi.get(i)(refcorE)
                be[i] += w[k] * iu * phii * leng
        return be
    

    elif e.bctype == -2:
        # sigma_T
        print(nt)
        N11 = np.zeros((Nloc, Nloc))
        E = meshelt[e.neighbor[0]]
        BE, bE = computeBE(meshvertex[E.vertex[0]], meshvertex[E.vertex[1]], meshvertex[E.vertex[2]])
        for k in range(NG):
            realcor = ref2I(p1, p2, s[k])
            refcorE = E2hat(BE, bE, realcor)
            for i in range(Nloc):
                phii = phi.get(i)(refcorE)
                for j in range(Nloc):
                    phij = phi.get(j)(refcorE)
                    N11[i][j] += phii * phij * w[k] * leng * nt
        return N11
    
Ntol = Nelt * Nloc
Fglobal = np.zeros(Ntol)
Aglobal = np.zeros((Ntol, Ntol))
for k in range(Nelt):
    E = meshelt[k]
    A, F = computeAF(E)
    for i in range(Nloc):
        ie = i + k * Nloc
        Fglobal[ie] += F[i]
        for j in range(Nloc):
            je = j + k * Nloc
            Aglobal[ie][je] += A[i][j]


for k in range(Nface):
    e = meshface[k]
    if e.bctype == 0:
        noe1 = e.neighbor[0]
        noe2 = e.neighbor[1]
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
                
    elif e.bctype == 1:
        noe1 = e.neighbor[0]
        M11, be = computeM(e)
        for i in range(Nloc):
            ie = i + noe1 * Nloc
            Fglobal[ie] += be[i]
            for j in  range(Nloc):
                je = j + noe1 * Nloc
                Aglobal[ie][je] += M11[i][j]


for k in range(Nface):
    e = meshface[k]
    nvec = normvec(e)
    nt = nvec[1]
    if e.bctype == 0:
        noe1 = e.neighbor[0]
        noe2 = e.neighbor[1]
        if nt > 1e-10:
            N11, N21 = computeN(e)
            for i in range(Nloc):
                ie = i + noe1 * Nloc
                for j in range(Nloc):
                    je = j + noe1 * Nloc
                    Aglobal[ie][je] += N11[i][j]
                    
            for i in range(Nloc):
                ie = i + noe2 * Nloc
                for j in range(Nloc):
                    je = j + noe1 * Nloc
                    Aglobal[ie][je] += N21[i][j]
        elif abs(nt) < 1e-10:
            pass
        elif nt < -1e-10:
            N12, N22 = computeN(e)
            for i in range(Nloc):
                ie = i + noe2 * Nloc
                for j in range(Nloc):
                    je = j + noe2 * Nloc
                    Aglobal[ie][je] += N22[i][j]
                    
            for i in range(Nloc):
                ie = i + noe1 * Nloc
                for j in range(Nloc):
                    je = j + noe2 * Nloc
                    Aglobal[ie][je] += N12[i][j]    
    elif e.bctype == -1:
        noe1 = e.neighbor[0]
        be = computeN(e)
        for i in range(Nloc):
            ie = i + noe1 * Nloc
            Fglobal[ie] += be[i]
            

    elif e.bctype == -2:
        noe1 = e.neighbor[0]
        N11 = computeN(e)
        for i in range(Nloc):
            ie = i + noe1 * Nloc
            for j in  range(Nloc):
                je = j + noe1 * Nloc
                Aglobal[ie][je] += N11[i][j]
    else:
        pass


u = np.linalg.solve(Aglobal, Fglobal)
u = u.reshape(Nelt, Nloc)
print("solution:", u)

def basisfunction(i, E: element, p):
    '''accept real coordinates and return phi_i^E(x) x \in E, defined by hat phi'''
    p1 = meshvertex[E.vertex[0]]; p2 = meshvertex[E.vertex[1]]; p3 = meshvertex[E.vertex[2]]
    BE, bE = computeBE(p1, p2, p3)
    refcor = E2hat(BE, bE, p)
    if is_in_ref(refcor) != -1:
        val = phi.get(i, "Invalid")(refcor)
        return val
    else:
        print('basis fun:?')
        return 0

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
    ax.view_init(elev=40, azim=220)
    n_inter = 10
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
    ax.set_title('Heat Equation')
    ax.set_box_aspect([1,1,1])

    filename = f'./pic/3D_eps{eps}sig{sigma}_para{para}.png'  
    plt.savefig ( filename )
    print ( '' )
    print ( '  Graphics saved as "%s"' % ( filename ) )
    plt.show()

def plot_heatmap():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_inter = 10
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
            sol[ie] = solution(u[i], i, real_p)-exact(real_p)
            plot_p[ie] = real_p
        
    ax.tricontourf(plot_p[:, 0], plot_p[:, 1], sol, cmap='jet')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Heat Equation Error Heatmap (2D)')
    plt.axis('equal')
    cbar = fig.colorbar(ax.collections[0], ax=ax, orientation='vertical')  

    filename = f'./pic/heatmap_eps{eps}sig{sigma}_para{para}.png'  
    plt.savefig ( filename )
    print ( '' )
    print ( '  Graphics saved as "%s"' % ( filename ) )
    plt.show()

plot3D()
plot_heatmap()
# plot_true_3D()
# plot_true_heat()