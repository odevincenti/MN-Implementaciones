########################################################################################################################
# Ecuaciones diferenciales a derivadas parciales
########################################################################################################################

import numpy as np

########################################################################################################################
# Ecuación de onda: u_tt = ku_xx
# k: constante de onda
# nro_hx: número de pasos en x
# nro_ht: numero de pasos en t
# u0: vector de N+1 valores con los valores de u(xi, 0)
# u0t: valor de u(0, t)
# uNt: valor de u(xN, t)
# hx: paso en x
# ht: paso en t
# Se asume ut(x, 0) = 0 y realiza una aproximacion de primer orden ui0 = ui1
# ----------------------------------------------------------------------------------------------------------------------
def wave(k, nro_hx, nro_ht, u0, u0t, uNt, hx, ht):

    # Matriz solucion
    u = np.zeros((nro_hx + 1, nro_ht + 1))
    u[:, 0] = u0                                # Primer columna
    u[0] = u0t                                  # Primer fila
    u[nro_hx] = uNt                             # Última fila
    # ut(x,0) = 0 => ui0 = ui1
    u[:, 1] = u[:, 0]
    # ( a1 a2 a3 a4 ) ( uij     )  = v.T * u
    #                 ( ui(j-1) )
    #                 ( u(i+1)j )
    #                 ( u(i-1)j )
    v = [2 - 2 * k * ht ** 2 / hx ** 2, -1, 4 * ht ** 2 / hx ** 2, 4 * ht ** 2 / hx ** 2]
    for j in range(1, nro_ht):
        for i in range(1, nro_hx):
            prev = [u[i, j], u[i, j - 1], u[i + 1, j], u[i - 1, j]]
            u[i, j + 1] = np.inner(v, prev)

    return u
# ----------------------------------------------------------------------------------------------------------------------
# EJEMPLO DE USO
k = 4                       # Cte de la ecuacion de onda
hx = 0.2                    # Paso en x
ht = 0.1                    # Paso en t
N = 5                       # Cantidad de pasos en x
M = 5                       # Cantidad de pasos en t

# Condiciones iniciales
# u(x,0) = sen(pi x)
u0 = [np.sin(np.pi * xi) for xi in np.linspace(0, N * hx, N + 1)]
# u(0,t) = u(1,t) = 0
u0t = 0
uNt = 0

u = wave(k, N, M, u0, u0t, uNt, hx, ht)
print(f"Matriz solucion:")
print(np.round(u, 3))
########################################################################################################################















