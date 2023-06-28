import numpy as np
import pandas as pd
from mpmath import *


mp.dps = 400


quantidade_de_bases = 3
quantidade_de_observacoes = 100  # 10, 30, 100
dimensoes = [4, 8, 16]
expoentes_fuzzy = [1.1, 8.325, 15.550, 22.775, 30.0]


"""
Lê os 3 resultados para cada inicialização e faz a média do índice de Hullermeier
"""


for dimensao in dimensoes:
    for expoente in expoentes_fuzzy:
        nome_pasta: str = f"resultados/obs_{quantidade_de_observacoes}/experimento_dimensao_{dimensao}_obs_{quantidade_de_observacoes}_expoente_fuzzy_{expoente}/"

        media_hullermeier = mpf(0)
        dict_hullermeier = {"Hullermeier 0": 0, "Hullermeier 1": 0, "Hullermeier 2": 0}
        for i in range(quantidade_de_bases):
            nome_arquivo: str = f"{nome_pasta}/experimento_inicializacao_{i}.csv"
            df = pd.read_csv(nome_arquivo, dtype=str)

            dict_hullermeier[f"Hullermeier {i}"] = df["hullermeier"][0]
            media_hullermeier += mpf(df["hullermeier"][0])

        df = pd.DataFrame(
            columns=[
                "quantidade de observações",
                "dimensão",
                "expoente fuzzy",
                "Hullermeier 0",
                "Hullermeier 1",
                "Hullermeier 2",
                "Média do Indice de Hullermeier",
            ]
        )

        nova_linha = {
            "quantidade de observações": quantidade_de_observacoes,
            "dimensão": dimensao,
            "expoente fuzzy": expoente,
            "Hullermeier 0": dict_hullermeier["Hullermeier 0"],
            "Hullermeier 1": dict_hullermeier["Hullermeier 1"],
            "Hullermeier 2": dict_hullermeier["Hullermeier 2"],
            "Média do Indice de Hullermeier": media_hullermeier / quantidade_de_bases,
        }

        df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
        df.to_csv(
            f"resultado_agregado/obs_{quantidade_de_observacoes}/experimento_dimensao_{dimensao}_obs_{quantidade_de_observacoes}_expoente_fuzzy_{expoente}.csv",
            encoding="utf-8",
            index=False,
        )


if __name__ == "__main__":
    ...
