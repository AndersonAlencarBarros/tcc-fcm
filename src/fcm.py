from matplotlib import pyplot as plt
import numpy as np
import random
import time
from numba import njit


@njit(cache = True)
def calculate_euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    """
        Calcula a distância Euclidiana entre dois pontos 
    """
    return np.sum((np.subtract(x, y)) ** 2)      


@njit(cache = True)
def matrix_norm(x: np.ndarray, y: np.ndarray) -> np.float64:
    return np.sqrt(np.sum((np.subtract(x, y)) ** 2)) 


@njit(cache = True)
def calculate_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
        Calcular distâncias de cada ponto a cada centro
    """
            
    distances: np.ndarray = np.array(
        [calculate_euclidean_distance(i, j) for i in data for j in centers]
    )
    
    """
        Cada linha representa um ponto, cada coluna representa um centro
        d[0][1] : distância do ponto zero ao centro um.
    """
    distances: np.ndarray = np.reshape(
            distances, (
                distances.shape[0] // centers.shape[0], 
                centers.shape[0]
            )
    )
    
    return distances


@njit(cache = True)
def update_membership(
    u: np.ndarray,
    data: np.ndarray,
    distances: np.ndarray, 
    n_clusters: int, 
    mu: np.float64, 
):
    
    CONST: np.float64 = (1 / (mu - 1))
    for i in range(data.shape[0]):
        for j in range(n_clusters):
            s = np.sum(
                np.array([
                    (distances[i][j] / distances[i][k]) ** CONST  for k in range(n_clusters)
                ]) 
            ) 
            
            u[j][i] = s ** -1
            

@njit(cache = True)
def update_centroids(u: np.ndarray, data: np.ndarray, mu: np.float64) -> np.ndarray:
    C = np.array(
            [np.sum((i ** mu) * j) / np.sum((i ** mu)) for i in u for j in data.T]
        )
        
    return np.reshape(C, (C.shape[0] // data.shape[1], data.shape[1]))


@njit(cache = True)
def mmg(
    u: np.ndarray, 
    data: np.ndarray, 
    centers: np.ndarray, 
    mu: np.float64
) -> np.float64:
    total = 0
    for i, d in enumerate(data):
        for j, c in enumerate(centers):
            distance = calculate_euclidean_distance(c, d)
            _u = u[j][i] ** mu
            
            total += (distance * _u)

    return total
 
 
class FCM():
    def __init__(self, n_clusters: int, mu: np.float64 = 2, eps=np.finfo(np.float64).eps):
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
            mu=self.mu
        )


    def _update_centroids(self):
        """
            Etapa de Minimização
            Atualização da posição dos centros
        """ 
        self.centers: np.ndarray = update_centroids(u=self.u, data=self.data, mu=self.mu)


    def J(self):
        """
            Mínimos Quadrados Generalizados (MMG)
        """
        j: np.float64 = mmg(
            u=self.u, 
            data=self.data, 
            centers=self.centers, 
            mu=self.mu
        )


    def fit(self, data: np.ndarray, u: np.ndarray):
        """
            Treinamento.
        """
        self.u = u
        self.data = data

        self._update_centroids()

        while True:
            u_copy = self.u.copy()

            self._update_membership()
            self._update_centroids()
            self.J()

            '''Critério de Parada'''
            if (matrix_norm(u_copy, self.u)) < self.eps:
                break


if __name__ == "__main__":
    
    n_clusters=2
    # n_samples=20000
    # dimensão=2 

    # np.random.seed(42)

    # X = np.random.normal((-1, 1), size=(n_samples, dimensão))

    X = np.array([
            [1, 3],
            [2, 5],
            [4, 8],
            [7, 9],
    ])
    
    u = np.array([
            [0.8, 0.7, 0.2, 0.1],
            [0.2, 0.3, 0.8, 0.9]
    ])
    
    """A soma dos graus de pertêncimento deve ser igual a 1"""
    # u = np.random.uniform(
    #     low=0, 
    #     high=1, 
    #     size=(n_clusters, len(X))
    # )
     
    # print(u)

    fcm = FCM(n_clusters=n_clusters, mu=2)
     
    start = time.perf_counter()
    fcm.fit(data=X, u=u)
    end = time.perf_counter()

    import skfuzzy as fuzz
    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=X, 
        c=n_clusters, 
        m=2, 
        error=np.finfo(np.float64).eps, 
        maxiter=30, 
        init=u.T
    )
    
    print()
    print(f"Elapsed = {end - start}s")
    print() 
    
    print('FCM')
    print('centers')
    print(fcm.centers)
    print('u')
    print(fcm.u)
    print()

    print(cntr)

    """Plot"""
    # fig, axes = plt.subplots(1, 2, figsize=(11,5))
    # axes[0].scatter(X[:,0], X[:,1], alpha=1)
    # axes[1].scatter(X[:,0], X[:,1], alpha=0.5)
    # axes[1].scatter(fcm.centers[:,0], fcm.centers[:,1], marker="+", s=1000, c='red')
    # plt.show()


'''
    Retornar:
    configuracao de entrada
    custo
    particao
    centros
'''

'''
    gerar incializacao aleatorias dentro das restricoes
    otimizar o metodo
'''
