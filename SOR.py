import numpy as np

#W debe estar entre 0 y 2 para que converga, si W=1 este este es el algoritmo de Gauss Seidel
def sorCuadrada(A,b,tol,maxiter,w=1):
    n=np.size(b)
    x=np.ones(n)
    for k in range(maxiter):
        for i in range(n):
            s=0
            for j in range(n):
                if i!=j:
                    s=s+A[i,j]*x[j]
            x[i]=(1-w)*x[i]+(w/A[i,i])*(b[i]-s)
    Return x
