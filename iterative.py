########################################################################################################################
# Métodos iterativos para la resolución de ecuaciones no lineales y sistemas lineales
########################################################################################################################


import numpy as np


########################################################################################################################
# bisec:
# Método de la biseccion en [a, b] con tolerancia 'tol', recibe callback a la funcion para evaluarla
# Converge sólo si hay un único cero, chequear
# Fórmula error total: |e_n|<=(b_0 - a_0)/2^(n+1) entonces el error del paso anterior es: |e_(n+1)|<=|e_n|/2 (CV lneal)
# ----------------------------------------------------------------------------------------------------------------------
def bisec(a, b, f, tol=np.finfo(float).eps, max_i=100):
    newa = a
    newb = b
    n = int(np.ceil(np.log2((newb - newa) / tol)) - 1)      # Calcula las iteraciones necesarias para el error pedido
    print(f"Se precisan {n} iteraciones")
    if n > max_i:
        print(f"El número de iteraciones necesario es mayor al máximo, se realizarán {max_i} iteraciones")
        n = max_i
    for i in range(n):
        c = newa + (newb - newa) / 2                        # Calcula c
        if (f(c) * f(newa)) >= 0:                           # Se fija ell signo de f(c)
            newa = c                                        # Si es igual al de a, a = c
        else:
            newb = c                                        # Si es igual al de b, b = c
        print(f"Raíz en iteración {i + 1}: {(newa + newb) / 2}")
    return (newa + newb) / 2
########################################################################################################################


########################################################################################################################
# punto_fijo:
# Método de punto fijo, arranca en xo con tolerancia 'tol', recibe callback a la función para evaluarla
# Condiciones para f:
# - f pertenece a C^1 en el intervalo
# - f(x) está en [a, b] cuando x pertenece a [a, b] (Mapeo) (Tip: GRAFICARLA)
# - Existe K tal que |f'(x)| <= K < 1 para todos los x en [a, b] (Derivada) (Tip: Mirar derivada segunda para justificar
# que es creciente o decreciente y mostrar que los extremos (a y b) pertenecen al intervalo)
# Fórmula error: |e_n| <= K^n * |e_0| entonces el error del paso anterior es: |e_(n+1)| <= K^n * |e_n| (CV lineal)
# ----------------------------------------------------------------------------------------------------------------------
def punto_fijo(xo, f, tol=np.finfo(float).eps, max_i=100):
    x0 = 0
    x1 = xo
    i = 0
    while np.abs(x1 - x0) > tol and i < max_i:
        x0 = x1
        x1 = f(x0)
        i += 1
        print(f"Raíz en iteración {i}: x{i} = {x1}")
    return x1
########################################################################################################################


########################################################################################################################
# newton_raphson:
# Método de Newton-Raphson empezando en x0 con error 'tol', recibe callback a la función y su derivada para evaluarlas
# Condiciones para f:
# - f' y f'' tienen signo constante en [a, b]
# Tip: Si tienen signo distinto, empiezo de la izquierda, si tienen el mismo signo, de la derecha
# Fórmula error: - Cero simple: |e_n| ~ |f''(x^*)/2f'(x^*)| * |e_(n-1)|^2 (CV cuadrática) para cada iteración
#                  Entonces: |e_n| ~ |f''(x^*)/2f'(x^*)|^(2^n - 1) * |e_0|^(2^n)
#                - Cero múltiple: CV lineal
# ----------------------------------------------------------------------------------------------------------------------
def newton_raphson(x0, f, fprime, tol=np.finfo(float).eps, max_i=100):
    x1 = x0
    i = 0
    fin = False
    for i in range(max_i):
        x0 = x1
        x1 = x0 - f(x0) / fprime(x0)
        print(f"Raíz en iteración {i}: x{i} = {x1}")
        if np.abs(x1 - x0) < tol:
            break
    return x1
########################################################################################################################


########################################################################################################################
# secante:
# Método de la secante empezando en x_2 y x_1 con error 'tol', recibe callback a la función para evaluarla
# Condiciones para f:
# Fórmula error: - Cero simple: |e_n| ~ |f''(x^*)/2f'(x^*)|^(R-1) * |e_(n-1)|^R para cada iteración
#                  Entonces: |e_n| ~ |f''(x^*)/2f'(x^*)|^(R^n - 1) * |e_0|^(R^n) con R = (1 + np.sqrt(5))/2
# ----------------------------------------------------------------------------------------------------------------------
def secante(x_2, x_1, f, tol=np.finfo(float).eps, max_i=100):
    i = 0
    fin = False
    while not fin:
        aux = x_1 - f(x_1) * (x_1 - x_2) / (f(x_1) - f(x_2))
        x_2 = x_1
        x_1 = aux
        i += 1
        print(f"Raíz en iteración {i}: x{i} = {x_1}")
        if np.abs(x_1 - x_2) < tol or i == max_i:
            fin = True
    return x_1
########################################################################################################################


########################################################################################################################
# xi: Arreglo de callbacks que reciben la iteracion actual y la anterior y devuelven el valor de las variables
def gauss_seidel(xi, xo, n):
    x = np.zeros((n, len(xo)))
    x[0] = xo
    for k in range(n):
        for j in range(len(xo)):
            x[k][j] = xi[j](x[k], x[k - 1])

    return x


"""
    No llegue a probar estas

# xi: Arreglo de callbacks que reciben la iteracion anterior y devuelven el valor de las variables
def jacobi(xi, xo, n):
    x = np.zeros((n, len(xo)))
    x[0] = xo
    for k in range(n):
        for j in range(len(xo)):
            x[k][j] = xi[j](x[k-1])

    return x

# xi: Arreglo de callbacks que reciben la iteracion actual y la anterior y devuelven el valor de las variables
def sor(xi, xo, n, w):
    x = np.zeros((n, len(xo)))
    x[0] = xo
    for k in range(n):
        for j in range(len(xo)):
            x[k][j] = (1-w)*x[k-1][j] + w*xi[j](x[k], x[k-1])

    return x

"""
########################################################################################################################
# épsilon de máquina: np.finfo(float).eps

print("\nBisección")
print(bisec(0, 1, lambda x: x - np.cos(x), max_i=100))
print("\nPunto Fijo")
print(punto_fijo(0.5, lambda x: np.cos(x), max_i=100))
print("\nNewton-Raphson")
print(newton_raphson(1, lambda x: x - np.cos(x), lambda x: 1 + np.sin(x), max_i=100))
print("\nSecante")
print(secante(0, 1, lambda x: x - np.cos(x), max_i=100))
