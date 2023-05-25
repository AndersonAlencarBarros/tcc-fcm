import numpy as np
from random import SystemRandom
from pathlib import Path
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
    for i, n in enumerate(n_clusters):
        for i in range(10):
            print(obs, n)
            u = np.random.uniform(
                size=(n, obs),
            )

            """ Normaliza por coluna, divide cada elemento de cada coluna pela soma total daquela coluna """
            u = u / np.sum(u, axis=0, keepdims=1)

            soma_colunas = np.sum(u, axis=0)

            """ Verificar cada coluna soma 1"""
            assert (
                np.allclose(soma_colunas, 1.0) == 1.0
            ), "Soma das colunas diferente de 1"

            dados = {"observacoes": obs, "n_cluster": n, "u": u.tolist()}

            base = Path(f"inicializacao/init_{obs}_{n}")
            jsonpath = base / f"init{i}.json"

            base.mkdir(exist_ok=True)
            jsonpath.write_text(dumps(dados))
