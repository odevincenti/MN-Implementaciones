import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def leastsq(A,b):
#     a=1
# #V,E,U = dvs(A,b)
#
#
# def dvs(A, b):
#     B = np.transpose(A) @ A
#     ava, V = np.linalg.eig(B)
#     E = np.zeros(np.shape(A))
#     U = np.zeros(np.shape(A))
#     for i in range(len(U[:, 0])):
#         E[i, i] = np.sqrt(ava[i])
#         if ava[i] > 0:
#             U[:, i] = A @ V[:, i]/E[i, i]

# Nota ultra importante, el rango de A tiene que ser igual a la dimension del conjunto Fil(A), basicamente, que tenga mas
# filas que columnas y sean un poco aleatorias
# En caso contrario no funciona
def leastsqrtrucho(A, b):
    Aspade = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A)    # Calculo A espada
    x = Aspade @ b                                                  # Calculo x: A espada . b
    return x

# A = np.array([[1.02, 1], [1.01, 1], [0.94, 1], [0.99, 1]])
# b = np.array([2.05, 1.99, 2.02, 1.93])

# print(np.dot(A.T, A))
# print(np.linalg.eig(np.dot(A.T, A))[0])
# print(np.linalg.eig(np.dot(A.T, A))[1])


# print(dvs(A))
# print("Lo que da:", leastsqrtrucho(A, b))
# print("Lo que debería dar:", np.linalg.lstsq(A, b, None)[0])

# EJEMPLO DEL PARCIAL
df = pd.read_csv("p53.csv")
x = np.array(df["x"].tolist())
y = np.array(df["y"].tolist())
n = x.shape[0]
fig, ax = plt.subplots()
ax.plot(x, y, ".", label="puntos")
ax.legend()


A = np.zeros((x.size, 3))
b = np.zeros(y.size)
A[:, 0] = np.sqrt(abs(x[:]))
A[:, 1] = np.cos(np.sqrt(abs(x[:])))
A[:, 2] = 1
print("Lo que da:", leastsqrtrucho(A, y))
print("Lo que debería dar:", np.linalg.lstsq(A, y, None)[0])

