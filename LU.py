def LU(mat):
    L = np.identity(mat[0].size)
    U = mat.astype(float)

    for i in range(mat[0].size - 1):
        m = U[:, i] / U[i][i]
        for j in range(i + 1, mat[0].size):
            L[j][i] = m[j]

            U[j] = U[j] - m[j] * U[i]
    return L, U
