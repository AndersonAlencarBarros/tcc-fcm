def _ler_json(caminho_arquivo: str):
    import json

    with open(caminho_arquivo) as arquivo:
        return json.load(arquivo)


def ler_base_de_dados(dimensao: int, observacoes: int):
    import numpy as np

    caminho_arquivo = f"../bases/dimensao_{dimensao}/obs_{observacoes}.json"

    dados = _ler_json(caminho_arquivo)
    return np.array(dados["bases"])


def ler_inicializacao(observacoes: int, n_clusters: int):
    import numpy as np

    caminho_arquivo = f"../inicializacao/init_{observacoes}/init_{n_clusters}.json"

    dados = _ler_json(caminho_arquivo)
    return np.array(dados["u"])


if __name__ == "__main__":
    from fcm import FCM
    import numpy as np
    import time
    import pandas as pd

    """
        mu ->  1,100 
                8,325
                15,550
                22,775
                30.000
                
                1 conjunto de dados
                10 inicializacoes para cada configuracao quantidade de observacoes e de agrupamentos
    """

    dimensoes = [2, 4, 8, 16]
    observacoes = [10, 100, 1000, 10000]
    qnt_agrupamentos = [
        [2, 4, 6, 9],
        [2, 34, 66, 99],
        [2, 334, 666, 999],
        [2, 3334, 6666, 9999],
    ]

    for d in dimensoes:
        print(f"dimensao {d}")

        for obs, n_clusters in zip(observacoes, qnt_agrupamentos):
            print(f"observacoes {obs}")

            base_de_dados = ler_base_de_dados(dimensao=d, observacoes=obs)

            df = pd.DataFrame(
                columns=[
                    "iteração",
                    "dimensão",
                    "mu",
                    "quantidade de observações",
                    "quantidade de agrupamentos",
                    "tempo de execução (s)",
                ]
            )

            for n in n_clusters:
                print(f"qnt_agrupamentos {n}")

                inicializacao = ler_inicializacao(observacoes=obs, n_clusters=n)

                for m in range(1, 36):
                    mu = 1.1**m

                    print(f"iteração {m} - mu {mu}")

                    fcm = FCM(n_clusters=n, mu=mu)

                    start = time.perf_counter()
                    fcm.fit(data=base_de_dados, u=inicializacao)
                    end = time.perf_counter()

                    elapsed = end - start

                    nova_linha = {
                        "iteração": m,
                        "dimensão": d,
                        "mu": mu,
                        "quantidade de observações": obs,
                        "quantidade de agrupamentos": n,
                        "tempo de execução (s)": elapsed,
                    }

                    df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

                    print("Salvando CSV...")
                    df.to_csv(
                        f"experimento_{obs}/experimento_dimensao_{d}_obs_{obs}.csv",
                        encoding="utf-8",
                        index=False,
                    )
