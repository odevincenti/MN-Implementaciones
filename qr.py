########################################################################################################################
# Se busca resolver el problema de cuadrados mínimos mediante descomposición QR
########################################################################################################################

import numpy as np

########################################################################################################################
# leastsqr:
# Resuelve el problema de cuadrados mínimos usando descomposición QR (argmin||A.x-b||)
# Recibe la matriz A y el vector b. Devuelve el vector x.
# Requisitos: - Que la cantidad de filas de A sea mayor o igual a la de columnas
#             - Que A sea de rango completo (columnas o filas LI)
#             - Que las dimensiones de A y b sean compatibles
# ----------------------------------------------------------------------------------------------------------------------
def leastsqr(A, b):

    # Factorización QR
    QR = desc_qr(A)                                     # Realizo la descomposición QR
    Q1 = QR[0]                                          # Recupero Q1 de m*n y ortonormal de A
    R1 = QR[1]                                          # Recupero R1 triangular superior de n*n

    # Resuelvo para encontrar x mínima                  # R1.x = Q^T.b
    C = np.dot(np.transpose(Q1), b)                     # C = Q1^T.b
    x = rev_subs(R1, C)                                 # R1.x = C

    return x
########################################################################################################################


########################################################################################################################
# desc_qr:
# Realiza la descomposición QR reducida usando Gram-Schmidt. Devuelve Q1 y R1
# Asume que se cumplen los requisitos de leastsqr
# ----------------------------------------------------------------------------------------------------------------------
def desc_qr(A):
    m = A.shape[0]                                         # Defino las dimensiones m y n
    n = A.shape[1]

    Q = np.zeros(shape=(m, n))                             # Creo las matrices base
    R = np.zeros(shape=(n, n))

    for i in range(n):
        p = A[:, i]
        for k in range(i):                  # Calculo las proyecciones (Q[:, k] nunca va a ser nula porque es ortogonal)
            p = p - (np.dot(A[:, i], Q[:, k])/np.linalg.norm(Q[:, k])) * Q[:, k]
        Q[:, i] = p/np.linalg.norm(p)                      # Normalizo la columna
        R[i, i] = np.linalg.norm(p)                        # Calculo la diagonal de R
        for j in range(i + 1, n):
            R[i, j] = np.dot(A[:, j], Q[:, i])             # Calculo la esquina superior de R

    return Q, R
########################################################################################################################


########################################################################################################################
# rev_subs:
# Resuelve un sistema por sustitución hacia atrás (A.x = y). Devuelve x
# La matriz A ingresada debe ser triangular superior, y debe ser un vector
# ----------------------------------------------------------------------------------------------------------------------
def rev_subs(A, y):
    m = A.shape[0]                                                          # Defino m
    x = np.zeros(shape=(m, 1))                                              # Crea el vector base
    for i in range(m):
        d = 0
        for j in range(i):
            d += A[m - i - 1, m - j - 1] * x[m - j - 1]                     # Calcula los términos conocidos
        x[m - i - 1] = (1/A[m - i - 1, m - i - 1])*(y[m - i - 1] - d)       # Calcula una de las componente

    return x
########################################################################################################################


########################################################################################################################
def triang_adelante(mat, b):
    y = np.zeros(b.size)
    N = len(b)

    for n in range(N):
        suma = mat[n] @ y
        y[n] = (b[n] - suma) / mat[n][n]
    return y

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




A = np.array([[1, 3, 6],
              [2, 4, 7],
              [8, 9, 56]])


b = np.array([1, 2, 10])

print(np.linalg.lstsq(A, b))
print(leastsqr(A, b))
