
import numpy as np


########################################################################################################################
# cholesky:
# Toma la matriz A de nxn simétrica definida positiva y devuelve una matriz G de nxn triangular inferior tal que G.G^T=A
# ----------------------------------------------------------------------------------------------------------------------
def cholesky(A):
    (n, m) = A.shape
    avas = np.linalg.eigvals(A)  # Si algun autovalor es <=0, da error
    for i in avas:
        if i < np.finfo(float).eps:
            print("La matriz no es definida positiva")
            return "ERROR"
        if not (A == A.T).all():
            print("La matriz no es simétrica")
            return "ERROR"
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sumsq = 0
                if i != 0:
                    for l in range(i):
                        sumsq += (G[i, l]) ** 2
                G[i, i] = np.sqrt((A[i, i] - sumsq))
            elif i > j:
                sumG = 0
                if j != 0:
                    for l in range(j):
                        sumG += G[i, l] * G[j, l]
                G[i, j] = (A[i, j] - sumG) / G[j, j]
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

    if b.shape[0] < 2:
        b = np.atleast_2d(b).T  # para convertir vectores 1-D de n a 2-d de nx1

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
        x[i][0] = (b[i][0] - A[i][:i] @ x[:i]) / A[i, i]
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
    B = np.dot(np.transpose(A), A)
    G = cholesky(B)
    C = np.dot(np.transpose(A), b)
    y = solve_trig(G, C)
    X = solve_trig(np.transpose(G), y)
    return X
########################################################################################################################

def test():
    # Normal
    A = np.array([[-1, -1], [1, 0], [-1, 1]])
    b = np.array([[1], [2], [3]])
    prueba = leastsq(A, b)
    print('Test 1 (Comparacion con ejercicio hecho en clase)')
    print('A:\n', A)
    print('b:\n', b)
    print('x min (a mano, hecho en clase):')
    print(np.array([[-2 / 3], [1]]))
    print('x min (funcion):')
    print(prueba)
    print('\n La prueba es exitosa si x min (funcion de Numpy) y x min (funcion nuestra) son iguales\n')

    # Random
    invalid = True
    while invalid:
        A = np.random.randint(-50, 50, (20, 5))
        b = np.random.randint(-50, 50, (20, 1))
        avas = np.linalg.eigvals(np.dot(np.transpose(A), A))  # Revisa que AT.A sea definida positiva
        for i in avas:
            if abs(i) != i or i <= np.finfo(float).eps:
                invalid = True
                print(i)
                break
            invalid = False

    prueba = leastsq(A, b)
    resultadoCompu = np.linalg.lstsq(A, b, rcond=None)[0]

    print('Test 2 (Comparacion con funcion de Cuadrados Minimos de Numpy)')
    print('A:\n', A)
    print('b:\n', b)
    print('x min (funcion de Numpy):')
    print(resultadoCompu)
    print('x min (funcion nuestra):')
    print(prueba)
    print('Diferencia entre resultados:')
    print(prueba - resultadoCompu)
    print(
        '\n La prueba es exitosa si los valores de Diferencia entre resultados son muy chicos (comparables o inferiores al epsilon de maquina)\n')

    # Indeterminado
    A = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1], [2]])

    print('Test 3 (Sistema Indeterminado por Dimensiones)')
    print('A:\n', A)
    print('b:\n', b)
    print('Resultado de la funcion:')
    try:
        prueba = leastsq(A, b)
    except ValueError:
        print("Se detecto el error de Sistema Indeterminado")
        print('\n Prueba exitosa\n')
    else:
        print("No se detecto el error de Sistema Indeterminado, la funcion retorna:")
        print(prueba)
        print('\n Prueba fallida\n')

        # Incompatible
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1], [2]])
    print('Test 4 (Sistema Incompatible por Dimensiones)')
    print('A:\n', A)
    print('b:\n', b)
    print('Resultado de la funcion:')
    try:
        prueba = leastsq(A, b)
    except ValueError:
        print("Se detecto el error de Sistema Incompatible")
        print('\n Prueba exitosa\n')
    else:
        print("No se detecto el error de Sistema Incompatible, la funcion retorna:")
        print(prueba)
        print('\n Prueba fallida\n')
    print('Fin del test')
    return "Taran!"

test()