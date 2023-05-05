import numpy as np
from random import SystemRandom
from pathlib import Path
import os
from json import dumps


sr = SystemRandom()


observacoes = [10, 100, 1000, 10000]
qnt_agrupamentos = [
    [2, 4, 6, 9],
    [2, 34, 66, 99],
    [2, 334, 666, 999],
    [2, 3334, 6666, 9999],
]


for obs, n_clusters in zip(observacoes, qnt_agrupamentos):
    for n in n_clusters: 
        for i in range(1, 101):
            print(obs, n)
            u = [
                np.random.uniform(
                    low=0, 
                    high=1, 
                    size=(n, obs)
                ) 
            ]

            ''' Normalizar por coluna, a soma deve ser igual a 1 '''
            u /= np.sum(u)
            
            ''' Verificar cada coluna soma 1'''
        
            dados = {
                "observacoes": obs,
                "n_cluster": n,
                "u": u.tolist()
            }
            
            base = Path(f'inicializacao/init_{obs}')
            jsonpath = base / f"{i}_init_{n}_{obs}.json"

            base.mkdir(exist_ok=True)
            jsonpath.write_text(dumps(dados))
