import numpy as np
from fcm import FCM
from utils import ler_base_de_dados, ler_inicializacao
from mpmath import mpf
import pandas as pd
import os


observacoes = 100
dimensao = [2, 4, 8, 16]
qnt_agrupamentos = [2, 34, 66, 99]
mu = [1.1, 8.325, 15.550, 22.775, 30.0]

nome_pasta: str = f"experimento_{observacoes}"
nome_arquivo: str = f"experimento_{observacoes}.csv"

df = pd.DataFrame(
    columns=[
        "dimensão",
        "mu",
        "quantidade de observações",
        "quantidade de agrupamentos",
        "custo",
        "u",
    ]
)

for d in dimensao:
    base_de_dados = ler_base_de_dados(dimensao=d, observacoes=observacoes)

    for n_clusters in qnt_agrupamentos:
        for m in mu:
            j = mpf("inf")
            u: np.ndarray = []

            for i in range(10):
                inicializacao = ler_inicializacao(
                    iteracao=i, observacoes=observacoes, n_clusters=n_clusters
                )

                fcm = FCM(n_clusters=n_clusters, mu=m)
                fcm.fit(data=base_de_dados, u=inicializacao)

                custo = fcm.J
                if custo < j:
                    print(f"dimensao {d} n_clusters {n_clusters} mu {m} iter {i}")
                    j = custo
                    u = fcm.u

            nova_linha = {
                "dimensão": d,
                "mu": m,
                "quantidade de observações": observacoes,
                "quantidade de agrupamentos": n_clusters,
                "custo": j,
                "u": u,
            }

            if not os.path.exists(nome_pasta):
                os.makedirs(nome_pasta)

            df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
            df.to_csv(
                f"{nome_pasta}/{nome_arquivo}",
                encoding="utf-8",
                index=False,
            )
