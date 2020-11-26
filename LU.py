def lu_decomposition(A):
    #"""(I | A) -> (L | A)"""
    m=3                    #Le asigno una valor cuaqluiera para que sea un float
    U = copy.copy(A)
    L = np.eye(len(A))
    #print(f'Inicio:\nL:\n{L}\nU:\n{U}')  # comentar si no se quiere ver cada paso de la descomposicion
    # k: numero de paso de la descomposicion
    for k in range(len(U)):
        # j: fila que se esta eliminando
        for j in range(k+1, len(U)):
            m = U[j][k]/U[k][k]
            for w in range(len(U[0])):
                U[j][w] = U[j][w] - m*U[k][w]
            L[j][k] = m
        #print(f'Paso {k + 1}:\nL:\n{L}\nU:\n{U}') # comentar si no se quiere ver cada paso de la descomposicion
    return (L, U)