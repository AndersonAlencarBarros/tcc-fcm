from celery import Celery
from decouple import config


app = Celery("experimento", broker=config("REDIS_URL"))
app.conf.update(worker_concurrency=3, consumer_timeout=31622400000)
app.conf.broker_transport_options = {"visibility_timeout": 60 * 60 * 5}


def salvar_resultados(dados):
    ...

quantidade_de_inicializacoes = 3

@app.task
def treinamento(
    iteracao: int,
    dimensao: int,
    quantidade_de_observacoes: int,
    quantidade_de_agrupamentos: int,
    expoente_fuzzy: int,
):
    import numpy as np
    from fcm import FCM
    from utils import ler_base_de_dados, ler_inicializacao, retorna_particao_base
    from mpmath import mpf
    import pandas as pd
    import os
    from hullermeier import hullermeier
    

    base_de_dados = ler_base_de_dados(
        iteracao=iteracao, dimensao=dimensao, observacoes=quantidade_de_observacoes
    ) 

    for i in range(quantidade_de_inicializacoes):
        inicializacao = ler_inicializacao(
            iteracao=i, observacoes=quantidade_de_observacoes, n_clusters=quantidade_de_agrupamentos
        )

        fcm = FCM(n_clusters=quantidade_de_agrupamentos, mu=expoente_fuzzy)
        fcm.fit(data=base_de_dados, u=inicializacao)
        
        particao_base = retorna_particao_base(
            quantidade_de_agrupamentos=quantidade_de_agrupamentos, quantidade_de_observacoes=quantidade_de_observacoes
        )
        
        indice_hullemeier = hullermeier(fcm.u, particao_base)

        df = pd.DataFrame(
            columns=[
                "dimensão",
                "expoente fuzzy",
                "quantidade de observações",
                "quantidade de agrupamentos",
                "custo",
                "hullermeier",
            ]
        )

        nova_linha = {
            "dimensão": dimensao,
            "expoente fuzzy": expoente_fuzzy,
            "quantidade de observações": quantidade_de_observacoes,
            "quantidade de agrupamentos": quantidade_de_agrupamentos,
            "custo": str(fcm.J),
            "hullermeier": str(indice_hullemeier),
        }
    
        nome_pasta: str = f"resultados/obs_{quantidade_de_observacoes}/experimento_dimensao_{dimensao}_obs_{quantidade_de_observacoes}_expoente_fuzzy_{expoente_fuzzy}/"

        if not os.path.exists(nome_pasta):
            os.makedirs(nome_pasta)

        nome_arquivo: str = f"experimento_inicializacao_{i}.csv"

        df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
        df.to_csv(
            f"{nome_pasta}/{nome_arquivo}",
            encoding="utf-8",
            index=False,
        )
