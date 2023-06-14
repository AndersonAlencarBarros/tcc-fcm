import numpy as np
from mpmath import mpf, fabs, fmul, fdiv, fsub


def hullermeier(U: np.ndarray, V: np.ndarray):

    N = U.shape[1]
    d = mpf(0.0)

    for i in range(N):
        for j in range(i, N):
            if j > i:
                Eu = fsub(1, fdiv(np.sum(np.abs(U[:, i] - U[:, j])), 2))
                Ev = fsub(1, fdiv(np.sum(np.abs(V[:, i] - V[:, j])), 2))
                
                d += fabs(fsub(Eu, Ev))

    d /= fdiv(fmul(N, (N - 1)), 2)
    return fsub(1, d)


if __name__ == "__main__":
    from utils import ler_inicializacao

    U = ler_inicializacao(iteracao=0, observacoes=10, n_clusters=2)
    V = ler_inicializacao(iteracao=1, observacoes=10, n_clusters=2)

    print(U.shape)
    print(V.shape)

    print()
    print(hullermeier(U, V))
