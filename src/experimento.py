from celery import Celery
from decouple import config

 
app = Celery("experimento", broker=config("REDIS_URL"))
app.conf.update(worker_concurrency=3, consumer_timeout=31622400000)
app.conf.broker_transport_options = {"visibility_timeout": 60 * 60 * 5}


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

    quantidade_de_inicializacoes = 3
    
    base_de_dados = ler_base_de_dados(
        iteracao=iteracao, dimensao=dimensao, observacoes=quantidade_de_observacoes
    )
    
    melhor_custo = mpf("inf")
    melhor_particao_calculada = []
    

    for i in range(quantidade_de_inicializacoes):
        inicializacao = ler_inicializacao(
            iteracao=i,
            observacoes=quantidade_de_observacoes,
            n_clusters=quantidade_de_agrupamentos,
        )

        fcm = FCM(n_clusters=quantidade_de_agrupamentos, mu=expoente_fuzzy)
        fcm.fit(data=base_de_dados, u=inicializacao)
        
        if fcm.J < melhor_custo:
            melhor_custo = mpf(fcm.J)
            melhor_particao_calculada = fcm.u
        
    particao_base = retorna_particao_base(
        quantidade_de_agrupamentos=quantidade_de_agrupamentos,
        quantidade_de_observacoes=quantidade_de_observacoes,
    )

    indice_hullemeier = hullermeier(melhor_particao_calculada, particao_base)

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

    nome_arquivo: str = f"experimento_base_{iteracao}.csv"

    df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    df.to_csv(
        f"{nome_pasta}/{nome_arquivo}",
        encoding="utf-8",
        index=False,
    )


if __name__ == "__main__":
    quantidade_de_bases = 3
    quantidade_de_observacoes = [10, 30, 100]
    quantidade_de_agrupamentos = 2
    dimensoes = [4, 8, 16]
    expoentes_fuzzy = [1.1, 8.325, 15.550, 22.775, 30.0]

    list_tasks = []
    for dimensao in dimensoes:
        for quantidade in quantidade_de_observacoes:
            for expoente in expoentes_fuzzy:
                for i in range(quantidade_de_bases):
                    print(f"dimensao - {dimensao} - quantidade - {quantidade} - expoente {expoente} - i = {i}")
                    
                    list_tasks.append(
                        treinamento.s(
                            iteracao=i,
                            dimensao=dimensao,
                            quantidade_de_observacoes=quantidade,
                            quantidade_de_agrupamentos=quantidade_de_agrupamentos,
                            expoente_fuzzy=expoente,
                        )
                    )

    from celery import group

    group_tasks = group(list_tasks)
    result = group_tasks.apply_async()
