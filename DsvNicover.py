import numpy as np


def dvs(A, b):
    B = np.transpose(A) @ A
    ava, V = np.linalg.eig(B)
    E = np.zeros(np.shape(A))
    U = np.zeros(np.shape(A))
    for i in range(len(U[:, 0])):
        E[i, i] = np.sqrt(ava[i])
        if ava[i] > 0:
            U[:, i] = A @ V[:, i]/E[i, i]


def leastsqrtrucho(A, b):
    Aspade = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A)
    x = Aspade @ b
    return x


A = np.array([[1.02, 1], [1.01, 1], [0.94, 1], [0.99, 1]])
b = np.array([2.05, 1.99, 2.02, 1.93])

# print(np.dot(A.T, A))
# print(np.linalg.eig(np.dot(A.T, A))[0])
# print(np.linalg.eig(np.dot(A.T, A))[1])


# print(dvs(A))
print("Lo que da:", leastsqrtrucho(A, b))
print("Lo que debería dar:", np.linalg.lstsq(A, b, None)[0])
