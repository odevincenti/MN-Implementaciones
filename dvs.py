########################################################################################################################
# Descomposición en valores singulares
########################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################################
# leastdvs:
# Resuelve el problema de cuadrados mínimos usando descomposición por valores singulares
# ----------------------------------------------------------------------------------------------------------------------
def leastdvs(A, b):
    vs, V, U = dvs(A)                                   # Obtengo la descomposición parcial
    x = 0
    for i in range(len(vs)):
        x += ((U[:, i])@b)*(V[:, i]/vs[i])              # Hago sumatoria
    return np.array(x)
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
    V = np.transpose(np.array(V))
    U = []
    i = 0
    while i < len(sigmas) and sigmas[i] > 0:    # Filtro sigmas inválidos
        U.append((A@(V[i].T))/sigmas[i])        # Calculo U (parcialmente)
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


########################################################################################################################
# EJEMPLO SIMPLE
'''A = np.array([[1.02, 1], [1.01, 1], [0.94, 1], [0.99, 1]])
b = np.array([2.05, 1.99, 2.02, 1.93])

print("Lo que da:", leastdvs(A, b))
print("Lo que debería dar:", np.linalg.lstsq(A, b, None)[0])'''
########################################################################################################################


########################################################################################################################
# RESOLUCIÓN EJERCICIO DEL PARCIAL
# Se tiene la función y = acos(x^2) + bsin(x) + c y los valores para x e y provistos por el archivo
# ----------------------------------------------------------------------------------------------------------------------
'''df = pd.read_csv("p43.csv")
x = np.array(df["x"].tolist())
y = np.array(df["y"].tolist())
n = x.shape[0]
fig, ax = plt.subplots()
ax.plot(x, y, ".", label="puntos")
ax.legend()

A = np.zeros((x.size, 3))                   # Armo matriz A
A[:, 0] = np.cos(np.sqrt(abs(x[:])))
A[:, 1] = np.sin(np.sqrt(abs(x[:])))
A[:, 2] = 1

ya = leastdvs(A, y)                         # Busco coeficientes que mejor ajustan a la función
print("Coeficientes:", ya)
# yr = np.linalg.lstsq(A, y, None)[0]
# print("Lo que debería dar:", yr)

# Armo función de ajuste
def f(x):
    return ya[0]*np.cos(np.sqrt(abs(x))) + ya[1]*np.sin(np.sqrt(abs(x))) + ya[2]
    # return yr[0]*np.cos(np.sqrt(abs(x))) + yr[1]*np.sin(np.sqrt(abs(x))) + yr[2]

x = range(-50, 50)                          # Valores del eje x que toma el gráfico
plt.plot(x, [f(i) for i in x])              # Grafico función ajustada
plt.show()'''
########################################################################################################################




