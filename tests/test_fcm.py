import numpy as np
import time
import random
import pytest
from src.fcm import FCM
from fcmeans import FCM as _FCM


@pytest.fixture
def observacoes():
    X = np.array([
            [1, 3],
            [2, 5],
            [4, 8],
            [7, 9],
    ])
    
    return X


@pytest.fixture
def inicializacao():
    u = np.array([
            [0.8, 0.7, 0.2, 0.1],
            [0.2, 0.3, 0.8, 0.9]
    ])
    
    return u


def test_FCM():
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

    fcm = FCM(n_clusters=2, mu=2)
     
    start = time.perf_counter()
    fcm.fit(data=X, u=u)
    end = time.perf_counter()

    print()
    print(f"Elapsed = {end - start}s")
    print()
    
    print('FCM')
    print('centers')
    print(fcm.centers)
    print('u')
    print(fcm.u)
    

def test_FCM_2():
    n_samples=20000
    n_clusters=2
    dimensão=2 

    np.random.seed(42)
    X = np.random.normal((-1, 1), size=(n_samples, dimensão))
  
    u = np.random.uniform(
        low=0, 
        high=1, 
        size=(n_clusters, len(X))
    )
    
    print(u)

    fcm = FCM(n_clusters=n_clusters, mu=2)
     
    start = time.perf_counter()
    fcm.fit(data=X, u=u)
    end = time.perf_counter()

    print()
    print(f"Elapsed = {end - start}s")
    print() 
    
    print('FCM')
    print('centers')
    print(fcm.centers)
    print('u')
    print(fcm.u)
    print()


def test_comparacao_com_outra_implementacao():
    n_clusters=2
    
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
    
    fcm = FCM(n_clusters=n_clusters, mu=2)
    fcm.fit(data=X, u=u)
  
    print('FCM')
    print(fcm.centers)

    print('_FCM')
    _fcm = _FCM(n_clusters=2, m=2, error=1e-09)
    _fcm.fit(X)
    print(_fcm.centers)
    
    assert _fcm.centers.shape == fcm.centers.shape
