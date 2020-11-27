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
# edo: EDO. Si es de segundo orden, x'=x[0] y x=x[1]
# t0: Inicio del intervalo de tiempo
# tf: Fin del intervalo de tiempo
# h: Paso
# ci: Condiciones iniciales. IMPORTANTE: Si la EDO es de orden 2 x0[0]=x'(0) y x0[1]=x(0)
# Devuelve arreglo con los valores de t y x que representan la función buscada
# ----------------------------------------------------------------------------------------------------------------------
def ruku4(edo, t0, tf, h, ci):
  time = np.arange(t0, tf + h, h)
  x = np.zeros((len(time), len(ci)))
  x[0] = ci
  fvec = np.zeros((4, len(ci)))
  for i, t in enumerate(time[:-1]):
    fvec[0] = edo(t, x[i])
    fvec[1] = edo(t + h / 2, x[i] + h * fvec[0] / 2)
    fvec[2] = edo(t + h / 2, x[i] + h * fvec[1] / 2)
    fvec[3] = edo(t + h, x[i] + h * fvec[2])
    x[i + 1] = x[i] + h * (np.dot(np.array([1, 2, 2, 1]), fvec) / 6)
  return time, x
########################################################################################################################


########################################################################################################################
# find_T: Encuentra el pseudoperíodo de una onda estacionaria
# y: Arreglo con las posiciones de la onda en función del tiempo
# t0: Punto a partir del cual se puede considerar a la onda estacionaria (fin del transitorio)
# h: Paso de t
# ----------------------------------------------------------------------------------------------------------------------
def find_T(y, t0, h):
    max = []
    for i in range(int(t0/h), len(y) - 1):
        if y[i-1] < y[i] > y[i+1]:
            max.append(i)
    dif = np.zeros(len(max) - 1)
    for i in range(len(max) - 1):
        dif[i] = np.abs(max[i+1] - max[i])

    return np.mean(dif)*h
########################################################################################################################


########################################################################################################################
# EJEMPLO DE USO RUKU 4
# Resuelve: x'' + 2(|x|-1)x' + x = 0 con las condiciones iniciales x(0)=1 y x'(0)=1
# ----------------------------------------------------------------------------------------------------------------------
'''def edo(t, x):
    return np.array([- 2*(np.abs(x[1])**4-1)*x[0] - x[1], x[0]])

x0 = np.array([1.0, 1.0])
t0 = 0
tf = 50
Emax = 1e-5
n = 4

# Para estimar el paso, tengo que estimar c, tomo un h cualquiera (que converja)
h1 = 0.1
t1, x1 = ruku4(edo, t0, tf, h1, x0)
t2, x2 = ruku4(edo, t0, tf, h1/2, x0)
c = np.abs(x2[4][1] - x1[2][1])/((1-1/2**n)*h1**n)          # Estimo c
print("c estimado:", c)
h = np.power(Emax/c, 1/n)                                   # Estimo paso
print("h estimado:", h)

t, x = ruku4(edo, t0, tf, h, x0)                            # Resuelvo el problema

# Cálculo del error
t_2, x_2 = ruku4(edo, t0, tf, h/2, x0)
Error = np.abs(x2[4][1] - x1[2][1])/(2**n - 1)              # Calculo error
print("Error:", Error)

plt.plot(t, x[:, 1])                                        # Grafico función
plt.title("Función")
plt.ylabel("x")
plt.xlabel("t")
plt.show()

plt.plot(t, x[:, 0])                                        # Grafico derivada
plt.title("Derivada")
plt.xlabel("t")
plt.ylabel("x")
plt.show()

print("Período:", find_T(x[:, 1], 3, h))                    # Busco período
'''
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



