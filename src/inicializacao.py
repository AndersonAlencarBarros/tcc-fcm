import numpy as np
from random import SystemRandom
from itertools import product
from json import dumps
import os
from pathlib import Path


sr = SystemRandom()
observacoes = [10, 100, 1000, 10000]
dimensoes = [2, 4, 8, 16]


for dimensao, obs in product(dimensoes, observacoes):
    u = np.random.uniform(
        low=0, 
        high=1, 
        size=(dimensao, obs)
    )
    
    u = u / np.sum(u)

    dados = {
        "dimensao": dimensao,
        "observacoes": obs,
        "u": u.tolist()
    }

    base = Path(f'inicializacao/init_{dimensao}')
    jsonpath = base / f"init_{dimensao}_{obs}.json"

    base.mkdir(exist_ok=True)
    jsonpath.write_text(dumps(dados))
