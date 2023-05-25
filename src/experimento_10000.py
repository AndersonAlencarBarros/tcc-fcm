import numpy as np
import time
import pandas as pd
from fcm import FCM
from utils import ler_base_de_dados, ler_inicializacao
import os


observacoes = 10000
dimensao = [2, 4, 8, 16]
qnt_agrupamentos = [2, 3334, 6666, 9999]
mu = [1.1, 8.325, 15.550, 22.775, 30.0]


df = pd.DataFrame(
    columns=[
        "dimensão",
        "mu",
        "quantidade de observações",
        "quantidade de agrupamentos",
        "tempo de execução (s)",
        "custo",
        "u",
        "centers",
    ]
)

for d in dimensao:
    base_de_dados = ler_base_de_dados(dimensao=d, observacoes=observacoes)
  
    for n_clusters in qnt_agrupamentos: 
            for m in mu: 
                
                for j in range(10): 

                    inicializacao = ler_inicializacao(
                        iteracao=j, observacoes=observacoes, n_clusters=n_clusters
                    )
                
                    print(f"dimensao {d} n_clusters {n_clusters} j {j} mu {m}")

                    fcm = FCM(n_clusters=n_clusters, mu=m)

                    start = time.perf_counter()
                    fcm.fit(data=base_de_dados, u=inicializacao)
                    end = time.perf_counter()

                    elapsed = end - start 

                    nova_linha = {
                        "dimensão": d,
                        "mu": m,
                        "quantidade de observações": observacoes,
                        "quantidade de agrupamentos": n_clusters,
                        "tempo de execução (s)": elapsed,
                        "custo": fcm.j,
                        "u": fcm.u,
                        "centers": fcm.centers,
                    }

                    df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    
                    nome_pasta = f"experimento_{observacoes}"

                    if not os.path.exists(nome_pasta):
                        os.makedirs(nome_pasta)

                    df.to_csv(
                        f"{nome_pasta}/experimento_dimensao_{d}_obs_{observacoes}.csv",
                        encoding="utf-8",
                        index=False,
                    )
