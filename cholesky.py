
import numpy as np


########################################################################################################################
# cholesky:
# Toma la matriz A de nxn simétrica definida positiva y devuelve una matriz G de nxn triangular inferior tal que G.G^T=A
# ----------------------------------------------------------------------------------------------------------------------
def cholesky(A):
    (n, m) = A.shape                #Guardo dimensiones de A
    avas = np.linalg.eigvals(A)     # Si algun autovalor es <=0, da error
    for i in avas:
        if i < np.finfo(float).eps:
            print("La matriz no es definida positiva")
            return "ERROR"
        if not (A == A.T).all():
            print("La matriz no es simétrica")
            return "ERROR"
    G = np.zeros((n, n))            # defino matriz G a calcular
    for i in range(n):
        for j in range(n):
            sum = 0
            if i == j:
                if i != 0:
                    for l in range(i):          # calculo de elementos en la diagonal de G
                        sum += (G[i, l]) ** 2
                G[i, i] = np.sqrt((A[i, i] - sum))
            elif i > j:
                if j != 0:
                    for l in range(j):
                        sum += G[i, l] * G[j, l]
                G[i, j] = (A[i, j] - sum) / G[j, j]     # calculo de elementos que no estan en la diagonal de G
    return G
########################################################################################################################


########################################################################################################################
# solve_trig:
# Toma una matriz triangular A (superior o inferior) de nxn y un vector de nx1 y resuelve el sistema
# ----------------------------------------------------------------------------------------------------------------------
def solve_trig(A, b):
    if A.shape[0] != A.shape[1] != b.shape[0]:
        print("La matriz ingresada no es cuadrada o no son compatibles las dimensiones de A y b")
        return "ERROR"


    b = np.atleast_2d(b).T          # convierto b a forma de vector vertical

    n = b.shape[0]
    x = np.zeros((n, 1))
    esInf = True
    for i in range(n - 1):  # para trabajar solo con triangular inferiores
        if not ((A[i][i + 1:] == np.zeros(n - i - 1)).all()):
            A = np.flip(A)
            b = np.flip(b)
            esInf = False
            break

    x[0][0] = b[0][0] / A[0, 0]
    for i in range(1, n):
        x[i][0] = (b[i][0] - np.dot(A[i][:i],x[:i])) / A[i, i]
    if esInf:
        return x
    return np.flip(x)
########################################################################################################################


########################################################################################################################
# leastsq:
# Resuelve el problema de los cuadrados mínimos usando la descomposición de Cholesky
# ----------------------------------------------------------------------------------------------------------------------
def leastsq(A, b):
    if A.shape[0] < A.shape[1]:
        print("Sistema indeterminado")
        return "ERROR"
    if A.shape[0] != b.shape[0]:
        print("Sistema incompatible")
        return "ERROR"
    B = np.dot(np.transpose(A), A)                  # B= At . A
    G = cholesky(B)                                 # B = G . Gt
    y = solve_trig(G, np.dot(np.transpose(A), b))   # G.y = At.b
    x = solve_trig(np.transpose(G), y)              # Gt.x = y
    return x
########################################################################################################################
A=np.array( [[1,3],
            [2,4],
             [8,9]])


b=np.array([1,2,10])

print(np.linalg.lstsq(A,b))
print(leastsq(A,b))