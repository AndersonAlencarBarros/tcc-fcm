import numpy as np
from mpmath import *


mp.dps = 400


def hullermeier(U: np.ndarray, V: np.ndarray):
    N = U.shape[1] 
    d = mpf(0)

    for i in range(N):
        for j in range(i, N):
            Eu = fsub(1, np.linalg.norm(U[:, i] - U[:, j], ord=1) / 2.0 )
            Ev = fsub(1, np.linalg.norm(V[:, i] - V[:, j], ord=1) / 2.0 )
            # Eu = fsub(mpf(1), fdiv(np.sum(np.abs(U[:, i] - U[:, j])), mpf(2)))
            # Ev = fsub(mpf(1), fdiv(np.sum(np.abs(V[:, i] - V[:, j])), mpf(2)))

            d += fabs(fsub(Eu, Ev))

    d /=  mpf((N * (N - 1)) / 2.0) 
    
    return fsub(1, d)


if __name__ == "__main__":
    from utils import ler_inicializacao
    from random import SystemRandom
    sr = SystemRandom()
        

    U = np.array([
        [1.0, 0, 0],
        [0, 1.0, 1.0]
    ]) 
    V = np.array([
        [1.0, 1, 0],
        [0,   0, 1]
    ]) 
    
    n = 16
    k = 10
    for _ in range(10000):
        U = [[sr.random() for _ in range(k)] for _ in range(n)]
        V = [[sr.random() for _ in range(k)] for _ in range(n)]
          
        U = U / np.sum(U, axis=0, keepdims=1)
        V = V / np.sum(V, axis=0, keepdims=1)
        
        h = hullermeier(U, V)
        
        assert h >= 0
    
    
    # V = np.array([
    #     [0.1, 0.2, 0.7],
    #     [0.3, 0.5, 0.2]
    # ])

    # print(U.shape)
    # print(V.shape)

    # print()
    # print(hullermeier(U, V))
