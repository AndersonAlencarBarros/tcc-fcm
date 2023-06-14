from celery import Celery


app = Celery(
    "tasks", 
    broker="redis://default:Tj1gqGMlyiavz6y6ELwivGQSNJIhzBM2@redis-11264.c74.us-east-1-4.ec2.cloud.redislabs.com:11264"
)
app.conf.update(worker_concurrency=3, consumer_timeout=31622400000)
app.conf.broker_transport_options = {'visibility_timeout': 60 * 60 * 5}   


@app.task
def treinamento(dimensao: int, observacoes: int, n_clusters: int, mu: int):
    import numpy as np
    from fcm import FCM
    from utils import ler_base_de_dados, ler_inicializacao
    from mpmath import mpf
    import pandas as pd
    import os

    nome_pasta: str = f"experimento_{observacoes}"

    base_de_dados = ler_base_de_dados(dimensao=dimensao, observacoes=observacoes)

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

    nome_arquivo: str = f"experimento_dim_{dimensao}_nc_{n_clusters}_mu_{mu}.csv"

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
