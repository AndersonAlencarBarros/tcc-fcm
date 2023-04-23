from src.fcm import FCM
import numpy as np
import time


def func(x):
    return x + 1


def test_answer():
    assert func(4) == 5


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