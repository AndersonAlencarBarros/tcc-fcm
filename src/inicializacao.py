import numpy as np
from random import SystemRandom
from itertools import product
from json import dumps
import os
from pathlib import Path
import pandas as pd


sr = SystemRandom()


observacoes = [10000] # 10, 100, 1000,
qnt_agrupamentos = [
    # [2, 3, 6, 9],
    # [2, 24, 48, 99],
    # [2, 249, 498, 999],
    [2, 2499, 4998, 9999],
]
dimensoes = [2, 4, 8, 16]


for obs, n_clusters in zip(observacoes, qnt_agrupamentos):
    for n in n_clusters: 
        print(obs, n)
        u = [
            np.random.uniform(
                low=0, 
                high=1, 
                size=(n, obs)
            ) for _ in range(100)
        ]

        u /= np.sum(u)
    
        dados = {
            "observacoes": obs,
            "u": u.tolist()
        }
        
        pd.DataFrame(dados).to_json(f"inicializacao/init_{obs}/init_{n}_{obs}.json")

        # base = Path(f'inicializacao/init_{obs}')
        # jsonpath = base / f"init_{n}_{obs}.json"

        # base.mkdir(exist_ok=True)
        # jsonpath.write_text(dumps(dados))
