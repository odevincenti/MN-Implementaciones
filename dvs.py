########################################################################################################################
# Descomposición en valores singulares
########################################################################################################################

import numpy as np

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
    for i in range((min(A.shape[0], A.shape[1]))):
        if lista[i][0] < 0:                                 # Filtro avas negativos
            pass
        else:
            vs.append(np.sqrt(lista[i][0]))                 # Calculo valores singulares
            V.append(normalize(lista[i][1]))                # Normalizo aves

    vs = np.array(vs)
    V = np.array(V)

    U = np.zeros((A.shape[0], A.shape[0]))
    for i in range(len(V)):
        U[i] = normalize(np.dot(A, V[i])/vs[i])           #ERA AVAS O SIGMAS?
    for j in range(len(V), len(U)):
        R = np.random.rand(A.shape[0])
        p = R
        for k in range(j):
            # u0 = U[k]
            # aux1 = np.dot(R, u0)
            # aux2 = np.linalg.norm(U[k])
            # aux3 = aux1/aux2
            # aux4 = np.dot(aux3, U[k])
            # p = p - aux4
            p = p - np.dot((np.dot(R, U[k]) / np.dot(U[k].T, U[k])), U[k])     # Busco vectores ortonormales para agregar a U
        U[j] = normalize(p)


    print("vs", vs)
    print("V", V)
    print("U", U.T)
    print("CERO", np.dot(U[0], U[1])) #, np.dot(U[0], U[2]), np.dot(U[2], U[1]))
    print("UNO", np.linalg.norm(U[0]), np.linalg.norm(U[1])) #, np.linalg.norm(U[2]))

    return
########################################################################################################################


########################################################################################################################
# gaussian_triangulation:
# Triangula la matriz dada usando el método de Gauss
# ----------------------------------------------------------------------------------------------------------------------
def gaussian_triangulation(A):
    T = np.array(A, float)
    for k in range(len(T)):                                     # k: numero de paso de la eliminación
        for i in range(k + 1, len(T)):                          # i: fila que se esta eliminando
            if T[k][k] != 0:                                    # Reviso que el pivote no sea cero
                m = T[i][k] / T[k][k]
                T[i] = T[i] - m * T[k]
            else:                                               # Si el pivote era cero:
                for j in range(k + 1, len(T)):
                    if T[j][k] != 0:                            # Busco una fila con esa columna != 0
                        T[[k, j]] = T[[j, k]]                   # Intercambio filas
                        break
    return T
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


########################################################################################################################
# det:
# Calcula el determinante de una matriz de nxn usando el método de eliminación de Gauss y multiplicando los valores de
# su diagonal
# ----------------------------------------------------------------------------------------------------------------------
def det(A):
    A = gaussian_triangulation(A)
    d = 1
    for i in range(len(A)):
        d = d * A[i][i]
    return d
########################################################################################################################

A = np.array([[1, 0, 0, 0], [1, 0, 0, 1]])
b = np.array([[1], [2], [3]])

dvs(A)

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

