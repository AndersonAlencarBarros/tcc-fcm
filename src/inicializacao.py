import numpy as np
from random import SystemRandom
from pathlib import Path
import os
from json import dumps
from sklearn import preprocessing
from math import ceil


sr = SystemRandom()


observacoes = [10, 100, 1000, 10000]
qnt_agrupamentos = [
    [2, 4, 6, 9],
    [2, 34, 66, 99],
    [2, 334, 666, 999],
    [2, 3334, 6666, 9999],
]


for obs, n_clusters in zip(observacoes, qnt_agrupamentos):
    for i, n in enumerate(n_clusters): 
        # for i in range(1, 101):
            print(obs, n)
            l = []
            for _ in range(obs):
                u = np.random.uniform(
                        size=(n),
                    )
                
                ''' Normalizar por coluna, a soma deve ser igual a 1 '''
                u /= np.sum(u)
                
                 
                assert round(np.sum(u)) == 1.0, 'Soma das colunas diferente de 1'
                
                l.append(u.tolist())
              
            l = np.array(l) 
            
            ''' Verificar cada coluna soma 1'''
        
            dados = {
                "observacoes": obs,
                "n_cluster": n,
                "u": l.T.tolist()
            }
            
            base = Path(f'inicializacao/init_{obs}')
            jsonpath = base / f"init_{i}_{n}.json"

            base.mkdir(exist_ok=True)
            jsonpath.write_text(dumps(dados))
