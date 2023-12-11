
from dolfin import * 
import numpy as np 
import matplotlib.pyplot as plt 


N = 10 
mesh = UnitIntervalMesh(N) 
p = 1 
V = FunctionSpace(mesh, "CG", p) 

def boundary(x, on_boundary): return on_boundary

bc = DirichletBC(V, Constant(0), boundary) 

u = TrialFunction(V) 
v = TestFunction(V) 

a = inner(grad(u), grad(v))*dx 
f = Constant(0)
L = f*v*dx  

A, _ = assemble_system(a, L, bc) 

invA = np.linalg.inv(A.array())
obs = invA[int(N/2),:]

print (np.min(obs), np.max(obs)) 

plt.plot(obs) 
plt.show()



