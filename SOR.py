import numpy as np
#Cuando w=1 es el metodo de Gauss seidel, W debe estar siempre entre 0 y 2
def sorCuadrada(A,b,maxiter=100,tol=np.finfo(float).eps,w=1):
    n=np.size(b)
    x=np.ones(n)
    for k in range(maxiter):
        for i in range(n):
            x0=np.array(x)
            s=0
            for j in range(n):
                if i!=j:
                    s=s+A[i,j]*x[j]
            x[i]=(1-w)*x[i]+(w/A[i,i])*(b[i]-s)
        print(f"Iteracion:",k,"/vector", x)
        if (np.linalg.norm(x-x0) <= tol):
            break
    return x

A= np.array([[5, 2, -1],
             [1, -5, -2],
              [1, -1, 3]])
b = np.array([4,4,4])

print(sorCuadrada(A,b,100,w=0.93))