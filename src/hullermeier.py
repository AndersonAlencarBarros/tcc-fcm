import numpy as np
from mpmath import mpf, fabs, fsum, fmul


def hullermeier(U: np.ndarray, V: np.ndarray):
    N = U.shape[1]
    d = 0.0
    
    for i in range(N):
        for j in range(i, N):
            if j > i:
                Eu = 1.0 - (np.sum(np.abs(U[: ,i] - U[: ,j])) / 2.0)
                Ev = 1.0 - (np.sum(np.abs(V[: ,i] - V[: ,j])) / 2.0)
                d += np.abs(Eu - Ev)
    
    d /= (N * (N - 1)) / 2.0

    return 1.0 - d


from utils import ler_inicializacao

U = ler_inicializacao(iteracao=0, observacoes=10, n_clusters=2)
V = ler_inicializacao(iteracao=1, observacoes=10, n_clusters=2)

print(U.shape)
print(V.shape)

print()
print(hullermeier(U, V))
