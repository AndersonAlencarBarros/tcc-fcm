import numpy as np
from numba import njit
from typing import Optional


@njit(cache=True)
def calculate_euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    """
    Quadrado da distância Euclidiana entre dois pontos
    """
    return np.sum((np.subtract(x, y)) ** 2.0)


@njit(cache=True)
def matrix_norm(x: np.ndarray, y: np.ndarray) -> np.float64:
    """
    Norma de Frobenius
    """
    return np.sqrt(np.sum((np.subtract(x, y)) ** 2.0))


@njit(cache=True)
def calculate_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Calcular distâncias de cada ponto a cada centro
    """

    distances: np.ndarray = np.array(
        [calculate_euclidean_distance(i, j) for i in data for j in centers], dtype="f8"
    )

    """
        Cada linha representa um ponto, cada coluna representa um centro
        d[0][1] : distância do ponto zero ao centro um.
    """
    distances: np.ndarray = np.reshape(
        distances, (distances.shape[0] // centers.shape[0], centers.shape[0])
    )

    return distances


def __verificar_soma_igual_a_1__(matriz: np.ndarray) -> bool:
    """
    Verificar se a soma de cada coluna é igual a 1
    """
    from math import isclose

    soma_colunas = np.sum(matriz, axis=0)

    assert (
        all(isclose(x, 1.0) for x in soma_colunas)
    ) == True, "Soma das colunas diferente de 1"


# @njit(cache = True)
def update_membership(
    u: np.ndarray,
    data: np.ndarray,
    distances: np.ndarray,
    n_clusters: int,
    mu: np.float64,
) -> None:
    expoente: np.float64 = 1.0 / (mu - 1.0)
    for k in range(data.shape[0]):
        for i in range(n_clusters):
            razao_entre_as_distancias: np.float64 = 0.0
            for j in range(n_clusters):
                razao_entre_as_distancias += (
                    distances[k][i] / distances[k][j]
                ) ** expoente

            u[i][k] = 1.0 / razao_entre_as_distancias

    # __verificar_soma_igual_a_1__(u)


@njit(cache=True)
def update_centroids(u: np.ndarray, data: np.ndarray, mu: np.float64) -> np.ndarray:
    C = np.array(
        [np.sum((i**mu) * j) / np.sum((i**mu)) for i in u for j in data.T],
        dtype="f8",
    )

    return np.reshape(C, (C.shape[0] // data.shape[1], data.shape[1]))


@njit(cache=True)
def mmg(
    u: np.ndarray, data: np.ndarray, centers: np.ndarray, mu: np.float64
) -> np.float64:
    total = 0
    for i, d in enumerate(data):
        for j, c in enumerate(centers):
            distance = calculate_euclidean_distance(c, d)
            _u = u[j][i] ** mu

            total += distance * _u

    return total


class FCM:
    def __init__(
        self,
        n_clusters: int,
        mu: np.float64 = 2,
        eps = 0.01
    ):
        self.n_clusters = n_clusters
        self.mu = mu
        self.eps = eps

    def _update_membership(self):
        """
        Etapa de Atribuição
        Atualização do grau de pertencimento
        """
        distances = calculate_distances(self.data, self.centers)
        update_membership(
            u=self.u,
            data=self.data,
            distances=distances,
            n_clusters=self.n_clusters,
            mu=self.mu,
        )

    def _update_centroids(self):
        """
        Etapa de Minimização
        Atualização da posição dos centros
        """
        self.centers: np.ndarray = update_centroids(
            u=self.u, data=self.data, mu=self.mu
        )

    def J(self):
        """
        Mínimos Quadrados Generalizados (MMG)
        """
        j: np.float64 = mmg(u=self.u, data=self.data, centers=self.centers, mu=self.mu)

        print(j)

    def _gerar_inicializacao(self) -> np.ndarray:
        u: np.ndarray = np.random.uniform(size=(self.n_clusters, self.data.shape[0]))

        """ Normaliza por coluna, divide cada elemento de cada coluna pela soma total daquela coluna """
        u = u / np.sum(u, axis=0, keepdims=1)

        # assert __verificar_soma_igual_a_1__(u) == True

        return u

    def fit(self, data: np.ndarray, u: Optional[np.ndarray] = None) -> None:
        """
        Treinamento.
        """
        self.data = data
        self.u = self._gerar_inicializacao() if u is None else u

        self._update_centroids()


        while True:
            u_copy: np.ndarray = self.u.copy()

            self._update_membership()
            self._update_centroids()


            """Critério de Parada"""
            if (matrix_norm(u_copy, self.u)) < self.eps:
                break


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import time
    from experimento import ler_base_de_dados, ler_inicializacao

    # n_clusters=2
    # mu = 1.61

    # X = np.array([
    #         [1, 3],
    #         [2, 5],
    #         [4, 8],
    #         [7, 9],
    # ])

    # u = np.array([
    #         [0.8, 0.7, 0.2, 0.1],
    #         [0.2, 0.3, 0.8, 0.9]
    # ])

    # fcm = FCM(n_clusters=n_clusters, mu=mu)

    # start = time.perf_counter()
    # fcm.fit(data=X, u=u)
    # end = time.perf_counter()

    from pprint import pprint

    base_de_dados = ler_base_de_dados(dimensao=2, observacoes=10)

    # pprint(base_de_dados)
    pprint(base_de_dados.shape)

    inicializacao = ler_inicializacao(observacoes=10, n_clusters=6)

    # pprint(inicializacao)
    pprint(inicializacao.shape)

    fcm = FCM(n_clusters=2, mu=1.61)

    start = time.perf_counter()
    fcm.fit(
        data=base_de_dados,
        u=inicializacao,
    )
    end = time.perf_counter()

    print()
    print(f"Elapsed = {end - start}s")
    print()

    print("FCM")
    print("centers")
    print(fcm.centers)
    print("u")
    print(fcm.u)
    print()

    """Plot"""
    # fig, axes = plt.subplots(1, 2, figsize=(11,5))
    # axes[0].scatter(X[:,0], X[:,1], alpha=1)
    # axes[1].scatter(X[:,0], X[:,1], alpha=0.5)
    # axes[1].scatter(fcm.centers[:,0], fcm.centers[:,1], marker="+", s=1000, c='red')
    # plt.show()


"""
    Retornar:
    configuracao de entrada
    custo
    particao
    centros
"""
