########################################################################################################################
# Optimización
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from copy import copy

########################################################################################################################
# steepest_descent: Busca el mínimo de la función f utilizando el método de máximo descenso
#   f: funcion a minimizar
#   vf: gradiente de f
#   xo: punto inicial del algoritmo
#   max_i: Número máximo de iteraciones
# ----------------------------------------------------------------------------------------------------------------------
def steepest_descent(f, vf, xo, tol=np.finfo(float).eps, max_i=100):
    x = xo
    for k in range(max_i):
        d = -vf(x)
        if d.all() < tol:       # Si d = (0, 0, 0) terminamos
            print(f"Se realizaron {k} iteraciones")
            break
        else:
            if np.count_nonzero(d) > 0:
                g = lambda alpha: f(x + alpha*d)
                alpha_min = argmin(g, 0, 1)
                x = x + alpha_min * d
        # print(f'Iteración {k + 1}')
        # print(f'd={d}')
        # print(f'x{k+1}={x}\n')

    return x
########################################################################################################################


########################################################################################################################
# qNewton: Implementa el algoritmo de cuasi-Newton BFGS
# Recibe: - f: Handle de la función a minimizar
#         - vf: Handle del gradiente de la función
#         - x0: Valor cercano al que minimiza la función
#         - tol: Tolerancia
#         - max_i: Número máximo de iteraciones
# ----------------------------------------------------------------------------------------------------------------------
def qNewton(f, vf, x0, tol=np.finfo(float).eps, max_i=100):
    x_1 = x0
    B = np.diag(np.full(vf(x0).shape, 0.1))             # Creo el B base
    i = 0
    fin = True

    while i < max_i and fin:
        x_0 = x_1                                       # Actualizo x
        d = -np.dot(B, vf(x_0))                         # Calculo d
        if np.linalg.norm(d) >= np.finfo(float).eps:
            g = lambda alpha: f(x_0 + alpha * d)
            a = argmin(g, 0, 1)                         # Calculo alfa que minimice f

            x_1 = x_0 + a * d                           # Calculo nueva x

            if np.linalg.norm(x_1 - x_0) >= tol and np.linalg.norm(d) >= np.finfo(float).eps:        # Me fijo si terminó
                s = a * d
                y = vf(x_1) - vf(x_0)
                B += ((s.T@y + (y.T@B)@y) * (s@s.T)) / (s.T@y) ** 2 - ((B@y)@s.T + s@(y.T@B)) / (s.T@y)  # Aplico BFGS
                i += 1
                # print(f"iteración {i}: min = {x_1}")
            else:
                fin = False                             # Si terminó, salgo del loop
        else:
            fin = False

    print(f"Se realizaron {i} iteraciones")

    return x_1
########################################################################################################################


########################################################################################################################
# argmin: Realiza la interpolación cuadrática de una función en R y encuentra el mínimo del polinomio
# Recibe: - f: Handle de la función a minimizar
#         - x: Punto de inicio de la minimización (Primer nodo)
#         - h: Estimación de la distancia entre los nodos
# Devuelve: Valor que minimiza el polinomio interpolador
# ----------------------------------------------------------------------------------------------------------------------
def argmin(f, x, h0):
    falta = True
    h = h0
    ya = f(x)
    yb = f(x + h)
    yc = f(x + 2 * h)
    while falta and h > 1e-10:                      # Se define h = 1e-10 como el menor h soportado
        if ya > yb > yc:            # Si no encuentra un mínimo,
            h = 2*h                 # Duplica el paso
        elif ya < yb < yc:          # Si se pasa del mínimo,
            h = h / 2               # Toma la mitad del paso
        else:
            falta = False           # Encontró un mínimo

        ya = f(x)
        yb = f(x + h)
        yc = f(x + 2*h)

    hmin = (4*yb - 3*ya - yc) / (4*yb - 2*ya - 2*yc) * h        # Calcula el mínimo del polinomio interpolador
    return x + hmin
########################################################################################################################


# Ej 6
print("Máximo descenso:", steepest_descent(lambda x: x**4, lambda x: 4*x**3, np.array([-2])))
print("Cuasi Newton:", qNewton(lambda x: x**4, lambda x: 4*x**3, np.array([-2]), 1e-2))




def nelder_mead_verbose(f, x_values, N):
    """
    nelder_mead_verbose:
    Minimiza la funcion f con el metodo de Nelder Mead, explicando
    paso a paso.
    @param f: funcion a minimizar, de n variables
    @param x_values: arreglo de n+1 puntos de dimension n
    @param N: numero de iteraciones
    @returns los poligonos formados en cada paso
    IMPORTANTE: x_values DEBE SER DE TIPO FLOAT O SINO REDONDEA
    """
    polygons = []
    for k in range(N):
        print(f'Inicio iteración {k + 1}: ')
        # Ordenar valores de x segun el valor de f(x)
        x_values = np.asarray(sorted(x_values, key=lambda x: f(x)))
        # Optimo: el x para el minimo valor
        O = x_values[0]
        print(f'1.a) O = {O} | f(O) = {f(O)}')
        # Bueno: el x para el segundo valor mas grande
        B = x_values[-2]
        print(f'1.b) B={B} | f(B)={f(B)}')
        # Peor: el x para el maximo valor
        P = x_values[-1]
        print(f'1.c) P={P} | f(P)={f(P)}')
        polygons.append(copy(x_values))
        M = np.mean(x_values[:-1], axis=0)
        print(f'2. Baricentro M = {M}')
        R = 2*M - P
        print(f'3. Reflexión R = {R} | f(R) = {f(R)}')
        if f(R) < f(O):
            print(f'4.1 Expansión: R mejor que O')
            E = 3*M - 2*P
            print(f'E = {E} | f(E) = {f(E)}')
            x_values[-1] = E if f(E) < f(O) else R
        elif f(R) > f(P):
            print(f'5.1 Contracción: R peor que P')
            C1 = (P + M) / 2
            print(f'C1 = {C1} | f(C1) = {f(C1)}')
            C2 = (R + M) / 2
            print(f'C2 = {C2} | f(C2) = {f(C2)}')
            C = C1 if f(C1) < f(C2) else C2
            print(f'C = {C} | f(C) = {f(C)}')               # El f(C) está sólo para ubicarme en lo que está pasando
            if f(C) < f(P):
                print(f'C mejor que P => P=C')
                x_values[-1] = C
            else:
                x_values[1:] = [(xi + O) / 2 for xi in x_values[1:]]
                print(f'5.2 Encogimiento: C peor que P, encogemos hacia O')
        else:
            print(f'Ni expansión ni contracción, P=R')
            x_values[-1] = R

        if len(x_values[0]) == 2:
            print(f'Triángulo iteración {k+1}: O = {x_values[0]} | B = {x_values[-2]} | P = {x_values[-1]} \n')
        else:
            print(f'Polígono iteración {k+1}:, del óptimo al peor, x{k+1}:\n {x_values}')
    # Reordeno y agrego ultimo polígono
    # Ordenar valores de x segun el valor de f(x)
    x_values = np.asarray(sorted(x_values, key=lambda x: f(x)))
    polygons.append(x_values)
    return polygons

'''
def plot_triangles(triangles, title):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for k in range(len(triangles)):
        plt.scatter(triangles[k, :, 0], triangles[k, :, 1], c=colors[k % len(colors)], label=f'Iteración {k}')
        t = plt.Polygon(triangles[k], color=colors[k % len(colors)], fill=False)
        plt.gca().add_patch(t)
    plt.legend()
    plt.title(title)
    plt.show()


# Ejercicio 7: Metodo de Nelder-Mead
# g(x,y) = (x-2)^2 + (y-1)^2

def g_ej7(X):
    x = X[0]
    y = X[1]
    return (x-2)**2 + (y-1)**2

# Numero de iteraciones pedidas
N = 4
xo = [0.0, 3]
x1 = [3, 0]
x2 = [3, 3]
triangles_ej7 = np.asarray(nelder_mead_verbose(g_ej7, [xo, x1, x2], N))
plot_triangles(triangles_ej7, 'Triángulos en cada iteración')

'''












