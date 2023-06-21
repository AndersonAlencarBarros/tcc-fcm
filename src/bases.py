from itertools import product
from random import SystemRandom
from json import dumps
import os
from pathlib import Path


sr = SystemRandom()


quantidade_de_bases = 3
observacoes = [10, 30, 100]
dimensoes = [4, 8, 16]


for dimensao, obs in product(dimensoes, observacoes):
    for i in range(quantidade_de_bases):
        base = [[sr.random() for _ in range(dimensao)] for _ in range(obs)]
          
        dados = {"dimensao": dimensao, "observacoes": obs, "base": base}
   
        base = Path(f"bases/dimensao_{dimensao}_obs_{obs}")
        jsonpath = base / f"base_{i}.json"

        base.mkdir(exist_ok=True)
        jsonpath.write_text(dumps(dados))
