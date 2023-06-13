from celery import Celery, group, chord
from time import sleep

app = Celery("tasks", broker="amqp://localhost")
app.conf.update(worker_concurrency=4)


@app.task
def add(x, y):
    sleep(15)
    return x + y
 

@app.task
def treinamento(dimensao: int, observacoes: int, n_clusters: int, mu: int):
    import numpy as np
    from fcm import FCM
    from utils import ler_base_de_dados, ler_inicializacao
    from mpmath import mpf
    import pandas as pd
    import os
    
    nome_pasta: str = f"experimento_{observacoes}"
   
    
    base_de_dados = ler_base_de_dados(
        dimensao=dimensao, observacoes=observacoes
    )
    
    j = mpf("inf")
    u: np.ndarray = []

    for i in range(10):
        inicializacao = ler_inicializacao(
            iteracao=i, observacoes=observacoes, n_clusters=n_clusters
        )

        fcm = FCM(n_clusters=n_clusters, mu=mu)
        fcm.fit(data=base_de_dados, u=inicializacao)

        custo = fcm.J
        if custo < j:
            print(f"dimensao {dimensao} n_clusters {n_clusters} mu {mu} iter {i}")
            j = custo
            u = fcm.u

    nome_arquivo: str = f"experimento_obs_{observacoes}_dim_{dimensao}_nc_{n_clusters}_mu_{mu}.csv"
    
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
    
    nova_linha = {
        "dimensão": dimensao,
        "mu": mu,
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
