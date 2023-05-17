from itertools import product
from random import SystemRandom
from json import dumps
import os
from pathlib import Path


sr = SystemRandom()


observacoes = [10, 100, 1000, 10000]
dimensoes = [2, 4, 8, 16]


for dimensao, obs in product(dimensoes, observacoes):
    # bases = []
    # for _ in range(100):
    vector = [ [sr.random() for _ in range(dimensao)] for _ in range(obs)]
    # bases.append(vector)

    dados = {
        "dimensao": dimensao,
        "observacoes": obs,
        "bases": vector
    }

    base = Path(f'bases/dimensao_{dimensao}')
    jsonpath = base / f"obs_{obs}.json"

    base.mkdir(exist_ok=True)
    jsonpath.write_text(dumps(dados))
    