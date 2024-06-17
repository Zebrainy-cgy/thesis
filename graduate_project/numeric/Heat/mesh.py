import numpy as np
class element:
    def __init__(self, vertex) -> None:
        self.face = np.zeros(3, dtype=int) # global number of faces
        self.parent = 0
        self.child = np.zeros(4, dtype=int)
        self.degree = 2 # polynomial degree
        self.reftype = -1 # -1 for inactive element and 0 for active element
        self.soldofs = []
        self.vertex = vertex # np.zeros(3, dtype=int)

class face:
    def __init__(self, vertex, neighbor) -> None:
        self.vertex = vertex # np.zeros(2, dtype=int) # global number of vertices
        self.neighbor = neighbor # np.zeros(2, dtype=int) # global number of vertices
        self.reftype = -1 # -1 for inactive element and 0 for active element
        self.bctype = 0 # -2 for t=T final boundary, -1 for initial boundary, 0 for interior face and 1 for Dirichlet face and 2 for Neumann face
