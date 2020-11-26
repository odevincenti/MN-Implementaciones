########################################################################################################################
# Ecuaciones diferenciales ordinarias
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################
# euler_nico:
# to: tiempo inicial
# f: callback a la funcion de carga, que debe tener el siguiente prototipo:
#    f(tn, yn) -> devuelve un elemento del mismo tipo y tamaño que yn (escalar o arreglo en caso de un sistema)
# yo: condiciones iniciales, en caso de ser un arreglo definen si se trata de una sola ecuación o un sistema
# h: paso
# n: número de iteraciones a realizar (implícitamente esto define el tf)
# ----------------------------------------------------------------------------------------------------------------------
def euler_nico(to, f, yo, h, n):
    if (type(yo) is int) or (type(yo) is float):
        yo = [yo]
    y = np.zeros((n+1, len(yo)))
    y[0] = yo
    t = to
    for k in range(1, n+1):
        y[k] = y[k-1] + h*f(t, y[k-1])
        t = t + h
    return y
########################################################################################################################


########################################################################################################################
# Implementación genérica de Euler (euler_f)
# f(t,x): derivada de x respecto al tiempo
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
########################################################################################################################
def euler_f(f, x0, t0, tf, h):
    N = int((tf - t0) / h)                      # número de puntos
    t = np.linspace(t0, tf, N + 1)
    n = x0.shape[0]                             # dimensión del problema
    x = np.zeros((n, N + 1))
    x[:, 0] = x0
    for k in range(N):
        x[:, k + 1] = x[:, k] + h * f(t[k], x[:, k])

    return t, x
########################################################################################################################


########################################################################################################################
# Implementación genérica de Taylor 2
# f(t,x): primera derivada de x respecto al tiempo
# g(t,x): segunda derivada de x respecto al tiempo
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
########################################################################################################################
def taylor2(f, g, x0, t0, tf, h):
    N = int((tf - t0) / h)                      # número de puntos
    t = np.linspace(t0, tf, N + 1)
    n = x0.shape[0]                             # dimensión del problema
    x = np.zeros((n, N + 1))
    x[:, 0] = x0
    i = h * h / 2.0
    for k in range(N):
        x[:, k + 1] = x[:, k] + h * f(t[k], x[:, k]) + i * g(t[k], x[:, k])

    return t, x
########################################################################################################################


########################################################################################################################
# wanted_step: calcula el paso requerido para un metodo de orden 'order' con constante 'c', un determinado 'error'.
# ----------------------------------------------------------------------------------------------------------------------
def wanted_step(c, error, order):
    return np.power(error/np.abs(c), 1.0/order)
########################################################################################################################


########################################################################################################################
# heun_nico:
# to: tiempo inicial
# h: paso
# n: numero de iteraciones a realizar (implicitamente esto define el tf)
# yo: condiciones iniciales, en caso de ser un arreglo definen si se trata de una sola ecuación o un sistema
# f: callback a la función de carga, que debe tener el siguiente prototipo:
#    f(tn, yn) -> devuelve un elemento del mismo tipo y tamaño que yn (escalar o arreglo en caso de un sistema)
# ----------------------------------------------------------------------------------------------------------------------
def heun_nico(to, h, n, f, yo):
    if (type(yo) is int) or (type(yo) is float):
        yo = [yo]
    y = np.zeros((n+1, len(yo)))
    y[0] = yo
    K1 = 0
    K2 = 0
    t = to
    for k in range(1, n+1):
        K1 = f(t, y[k-1])
        t = t + h
        K2 = f(t, y[k-1] + h*K1)
        y[k] = y[k-1] + h/2*(K1 + K2)
    return y
########################################################################################################################


########################################################################################################################
# Implementación genérica de Heun (heun_f)
# f(t,x): derivada de x respecto al tiempo
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
########################################################################################################################
def heun_f(f, x0, t0, tf, h):
    N = int((tf - t0) / h)                  # número de puntos
    t = np.linspace(t0, tf, N + 1)
    n = x0.shape[0]                         # dimensión del problema
    x = np.zeros((n, N + 1))
    x[:, 0] = x0
    for k in range(N):
        f1 = h * f(t[k], x[:, k])
        f2 = h * f(t[k] + h, x[:, k] + f1)
        x[:, k + 1] = x[:, k] + (f1 + f2) / 2.0

    return t, x
########################################################################################################################


########################################################################################################################
# Implementación genérica de Cauchy
# f(t,x): derivada de x respecto al tiempo
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
########################################################################################################################
def cauchy(f, x0, t0, tf, h):
    N = int((tf - t0) / h)  # número de puntos
    t = np.linspace(t0, tf, N + 1)
    n = x0.shape[0]  # dimensión del problema
    x = np.zeros((n, N + 1))
    x[:, 0] = x0
    h2 = h / 2.0
    for k in range(N):
        x[:, k + 1] = x[:, k] + h * f(t[k] + h2, x[:, k] + h2 * f(t[k], x[:, k]))

    return t, x
########################################################################################################################


########################################################################################################################
# Implementación genérica de Euler explícito
# f(t,x): derivada de x respecto al tiempo
# j(t,x): jacobiano de f(t,x) respecto a x
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
########################################################################################################################
def euleri(f, jf, x0, t0, tf, h):
    N = int((tf - t0) / h)  # número de puntos
    t = np.linspace(t0, tf, N + 1)
    n = x0.shape[0]  # dimensión del problema
    x = np.zeros((n, N + 1))
    x[:, 0] = x0
    i = np.eye(n)
    for k in range(N):
        y = x[:, k] + h * f(t[k], x[:, k])  # inicializo con Euler
        # resuelvo con Newton-Raphson: tolerancia = 1e-13
        for m in range(1000):
            delta = np.linalg.solve(i - h * jf(t[k] + h, y), -(y - x[:, k] - h * f(t[k] + h, y)))
            y = y + delta
            if np.linalg.norm(delta) < n * 1e-15:
                break

        x[:, k + 1] = y

    return t, x
########################################################################################################################


########################
# EJEMPLO
########################
R = 1e3	            #Valor de la resistencia
C = 1e-6	        #Valor de la capacidad
w = 2.0*np.pi*1000     #frecuencia angular de la señal de entrada
A = 1.0		        #amplitud de la señal de entrada
T = 5*2*np.pi/w	    #simulo cinco ciclos


def f(t, x):
    return (-21*x + np.e**(-t))

def j(t, x):
    return -21

t, y = euleri(f, j, np.array([0]), 0, 5, 0.1)
print(t, y)
plt.plot(t, y[0, :])
plt.show()

