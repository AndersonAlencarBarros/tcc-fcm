import numpy as np
from random import SystemRandom
from pathlib import Path
from json import dumps


sr = SystemRandom()

quantidade_de_inicializacoes = 3
observacoes = [10, 30, 100]
qnt_agrupamentos = 2


for obs in observacoes:
    for i in range(quantidade_de_inicializacoes):
        u = np.random.uniform(
            size=(qnt_agrupamentos, obs),
        )

        """ Normaliza por coluna, divide cada elemento de cada coluna pela soma total daquela coluna """
        u = u / np.sum(u, axis=0, keepdims=1)

        soma_colunas = np.sum(u, axis=0)

        """ Verificar cada coluna soma 1"""
        assert np.allclose(soma_colunas, 1.0) == 1.0, "Soma das colunas diferente de 1"

        dados = {"observacoes": obs, "u": u.tolist()}

        base = Path(f"inicializacao/init_{obs}")
        jsonpath = base / f"init{i}.json"

        base.mkdir(exist_ok=True)
        jsonpath.write_text(dumps(dados))
