
import numpy as np
import matplotlib.pyplot as plt
from copy import copy


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














