from utils import *
# basis functions are all defined on reference cell
# 1
def phi0(p): 
    if is_in_ref(p) != -1: return 1
    else: return 0
# x
def phi1(p): 
    if is_in_ref(p) != -1: return p[0]
    else: return 0
# y
def phi2(p): 
    if is_in_ref(p) != -1: return p[1]
    else: return 0
# x^2
def phi3(p): 
    if is_in_ref(p) != -1: return p[0] * p[0]
    else: return 0
# xy
def phi4(p): 
    if is_in_ref(p) != -1: return p[0] * p[1]
    else: return 0
# y^2
def phi5(p): 
    if is_in_ref(p) != -1: return p[1] * p[1]
    else: return 0
# (grad)basis functions
# (0,0)
def grad_phi0(p): 
    if is_in_ref(p) != -1: return np.array([0.0, 0.0])
# (1,0)
def grad_phi1(p): 
    if is_in_ref(p) != -1: return np.array([1.0, 0])
    else: 
        print('error')
        return np.zeros(2)
# (0,1)
def grad_phi2(p): 
    if is_in_ref(p) != -1: return np.array([0, 1.0])
    else: 
        print('error')
        return np.zeros(2)
# (2x,0)
def grad_phi3(p): 
    if is_in_ref(p) != -1: return np.array([2 * p[0], 0])
    else: 
        print('error')
        return np.zeros(2)
# (x,y)
def grad_phi4(p): 
    if is_in_ref(p) != -1: return np.array([p[1], p[0]])
    else: 
        print('error')
        return np.zeros(2)
# (0,2y)
def grad_phi5(p): 
    if is_in_ref(p) != -1: return np.array([0, 2 * p[1]])
    else: 
        print('error')
        return np.zeros(2)