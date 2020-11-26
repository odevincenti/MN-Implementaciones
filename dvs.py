########################################################################################################################
# Descomposición en valores singulares
########################################################################################################################

import numpy as np

########################################################################################################################
# leastdvs:
# Resuelve el problema de cuadrados mínimos usando descomposición por valores singulares
# ----------------------------------------------------------------------------------------------------------------------
def leastdvs(A, b):
    vs, V, U = dvs(A)
    x = 0
    for i in range(len(vs)):
        x += ((U[:, i])@b)*(V[:, i]/vs[i])
    return x
########################################################################################################################


########################################################################################################################
# dvs:
# Descompones la matriz A y encuentra sus valores singulares
# ----------------------------------------------------------------------------------------------------------------------
def dvs(A):
    B = np.dot(A.T, A)
    avas, aves = np.linalg.eig(B)               # Calculo autovalores y autovectores
    lista = list(zip(avas, aves))               # Ordeno los avas en orden descendiente y sus aves correspondientes
    lista.sort(key=lambda tup: tup[0], reverse=True)
    vs = []
    V = []
    U = []
    i = 0
    while i < min(A.shape[0], A.shape[1]) and lista[i][0] > 0:      # Filtro avas negativos
        vs.append(np.sqrt(lista[i][0]))                             # Calculo valores singulares
        V.append(normalize(lista[i][1]))                            # Normalizo aves
        U.append(normalize(np.dot(A, V[i]/vs[i])))                  # Calculo las primeras columnas de U
        i += 1

    vs = np.array(vs)
    V = np.transpose(np.array(V))
    U = np.transpose(np.array(U))
    # S = np.diag(vs)

    return vs, V, U
########################################################################################################################


########################################################################################################################
# normalize:
# Devuelve la versión normalizada del vector ingresado
# ----------------------------------------------------------------------------------------------------------------------
def normalize(v):
    if np.linalg.norm(v) != 0:
        v = v/np.linalg.norm(v)
    return v
########################################################################################################################


A = np.array([[1.02, 1], [1.01, 1], [0.94, 1], [0.99, 1]])
b = np.array([2.05, 1.99, 2.02, 1.93])

# print(np.dot(A.T, A))
# print(np.linalg.eig(np.dot(A.T, A))[0])
# print(np.linalg.eig(np.dot(A.T, A))[1])


# print(dvs(A))
print("Lo que da:", leastdvs(A, b))
print("Lo que debería dar:", np.linalg.lstsq(A, b, None)[0])

# vs = np.array([0, 1, 2, -3, 4, -5, -6, 7, 8])
# vsp = []
# for i in range(len(vs)):
#     if vs[i] < 0:
#         pass
#     else:
#         vsp.append(vs[i])
#
# print("vs:", vs)
# print("vsp", vsp)

