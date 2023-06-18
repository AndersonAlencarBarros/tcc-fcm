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
    
    n = 16 # Cluster   
    k = 10 # Dimensao
    for _ in range(10000):
        U = np.array([[sr.random() for _ in range(k)] for _ in range(n)])
        V = np.full((n, k), 1/n)
          
        U = U / np.sum(U, axis=0, keepdims=1)
        
        h = hullermeier(U, V)
        
        assert h >= 0
     
