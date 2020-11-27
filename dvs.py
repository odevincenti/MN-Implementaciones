########################################################################################################################
# Descomposición en valores singulares
########################################################################################################################

import numpy as np

########################################################################################################################
# leastdvs:
# Resuelve el problema de cuadrados mínimos usando descomposición por valores singulares
# ----------------------------------------------------------------------------------------------------------------------
def leastdvs(A, b):
    vs, V, U = dvs(A)                                   # Obtengo la descomposición parcial
    x = 0
    for i in range(len(vs)):
        x += ((U[:, i])@b)*(V[:, i]/vs[i])              # Hago sumatoria
    return x
########################################################################################################################


########################################################################################################################
# dvs:
# Descompones la matriz A (parcialmente) y encuentra sus valores singulares
# ----------------------------------------------------------------------------------------------------------------------
def dvs(A):
    B = A.T@A
    sigmas, V = np.linalg.eig(B)                # Calculo avas y aves
    idx = sigmas.argsort()[::-1]                # Ordeno avas
    sigmas = np.sqrt(sigmas[idx])               # Calculo sigmas ordenados
    V = V[:, idx]                               # Ordeno aves
    U = []
    i = 0
    while i < len(sigmas) and sigmas[i] > 0:    # Filtro sigmas inválidos
        U.append((A@np.atleast_2d(V[i]).T/sigmas[i]).flatten())         # Calculo U (parcialmente)
        i += 1
    U = np.transpose(np.array(U))
    return sigmas, V, U
########################################################################################################################


########################################################################################################################
# svd_complete:
# Descompone la matriz dada por sus valores singulares de forma completa
# Devuele E, V y U en ese orden
# ----------------------------------------------------------------------------------------------------------------------
def svd_complete(A):
    B = A.T@A
    sigmas, V = np.linalg.eig(B)
    r = len(sigmas)
    idx = sigmas.argsort()[::-1]
    sigmas = np.sqrt(sigmas[idx])
    V = V[:, idx]
    U = np.zeros((np.shape(A)[0], np.shape(A)[0]))
    for i, sigma in enumerate(sigmas):
        U[:, i] = (A@np.atleast_2d(V[i]).T/sigma).flatten()
    for i in range(r, np.shape(U)[0]):
        U[:, i] = np.random.random((np.shape(U)[0], 1)).flatten()
        for j in range(i):
            U[:, i] -= (U[:, j]@U[:, i])*U[:, j]
        U[:, i] = U[:, i]/np.linalg.norm(U[:, i])
    E = np.append(np.diag(sigmas), np.zeros((np.shape(A)[0]-len(sigmas), len(sigmas))), axis=0)
    return E, V, U
########################################################################################################################

A = np.array([[1.02, 1], [1.01, 1], [0.94, 1], [0.99, 1]])
b = np.array([2.05, 1.99, 2.02, 1.93])

print("Lo que da:", leastdvs(A, b))
print("Lo que debería dar:", np.linalg.lstsq(A, b, None)[0])



