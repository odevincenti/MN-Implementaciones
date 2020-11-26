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
# Implementación de ruku4
#
#
# ----------------------------------------------------------------------------------------------------------------------
def ruku4(f, t0, tf, h, x0):
  ts = np.arange(t0, tf + h, h)
  x = np.zeros((len(ts), len(x0)))
  x[0] = x0
  fvec = np.zeros((4, len(x0)))
  for i, t in enumerate(ts[:-1]):
    fvec[0] = f(t, x[i])
    fvec[1] = f(t + h / 2, x[i] + h * fvec[0] / 2)
    fvec[2] = f(t + h / 2, x[i] + h * fvec[1] / 2)
    fvec[3] = f(t + h, x[i] + h * fvec[2])
    x[i + 1] = x[i] + h * (np.dot(np.array([1, 2, 2, 1]), fvec) / 6)
  return ts, x
########################################################################################################################

def f(t, x):
    return np.array([1 - 2*(x[1]**4-1)*x[0] - x[1], x[0]])


ci = np.array([1.5, 1])
t0 = 0
tf = 75
h = 0.1
t, x = ruku4(f, t0, tf, h, ci)

plt.plot(t, x[:, 0])
plt.title("Función")
plt.ylabel("x")
plt.xlabel("t")
plt.show()

plt.plot(t, x[:, 1])
plt.title("Derivada")
plt.ylabel("x")
plt.xlabel("t")
plt.show()


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



