
from dolfin import * 
import numpy as np 
import matplotlib.pyplot as plt 

def S(x, alpha): 
    new_x = x.copy()
    new_x[:] = 0.0 
    for i in range(1,len(x)-1): # hack FEniCS includes dofs related to the boundary, making lmbda zero at the boundary therefore first and last item   
      if (x[i] >= alpha[i]): 
          new_x[i] = x[i] - alpha[i]  
      elif (x[i] <= - alpha[i]): 
          new_x[i] = x[i] + alpha[i] 
      else: 
          new_x[i] = 0  
    return new_x 




def sigma_max_and_AA(A): 
    AA = np.matmul(A.transpose(),A)/A.shape[0]
    e, v = np.linalg.eig(AA) 
    e.sort()
    return e[-1], AA 
    


def boundary(x, on_boundary): return on_boundary

def make_matrix(N, p, obs_indices): 
    mesh = UnitIntervalMesh(N) 
    V = FunctionSpace(mesh, "CG", p) 
    bc = DirichletBC(V, Constant(0), boundary) 
    u = TrialFunction(V) 
    v = TestFunction(V) 
    a = inner(grad(u), grad(v))*dx 
    f = Constant(0)
    L = f*v*dx  
    A, _ = assemble_system(a, L, bc) 
    invA = np.linalg.inv(A.array())
    obs = invA[indices,:]
    return obs 



N = 30 
p = 1 
index = int(N/2)
indices = [index]

A = make_matrix(N, p, indices) 

print (A)
print (np.min(A), np.max(A)) 

sigma, AA = sigma_max_and_AA(A)
print ("sigma ", sigma) 


nn = A.shape[1]
mm = A.shape[0]
x = np.random.random(nn)
xp = np.random.random(nn)
y = np.ones(mm)
#lmbda = A.flatten()
#print (lmbda) 
eps = 1e-3 
lmbda = eps * np.ones(nn) / nn 


it = 0 
max_it = 50000 
residuals = []
L1_norm_weigheds = []
while it <= max_it:  
    z = xp - 1.0/(sigma* nn)*(np.dot(AA, xp) - np.dot(A.transpose(), y))
    x = S(z, lmbda/sigma)   
    residual = np.dot(A, x) - y 
    residuals.append(np.linalg.norm(residual)) 
    L1_norm_weighed = np.inner(np.abs(lmbda), np.abs(x)) 
    L1_norm_weigheds.append(L1_norm_weighed) 
    print ("norm of residual, L1 norm, update ", np.linalg.norm(residual),  np.linalg.norm(x-xp), L1_norm_weighed) 
    print (np.linalg.norm(A,2))
    xp[:] = x[:]
    it = it+1 




plt.plot(x) 
plt.show()

plt.loglog(residuals) 
plt.loglog(L1_norm_weigheds) 
plt.legend(["equation residual ", "L1 comp"])
plt.show()


