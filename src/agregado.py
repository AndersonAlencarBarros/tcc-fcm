import glob
import pandas as pd
import os

observacoes = [10, 30, 100]
dimensoes = [4, 8, 16]
expoentes_fuzzy = [1.1, 8.325, 15.550, 22.775, 30.0]


"""
Gera o Resultado Agregado

Lê todos os resultados e concatena em um único csv.
"""


lista_dataframes = []
for obs in observacoes:
    for dimensao in dimensoes:
        for expoente in expoentes_fuzzy:
            caminho_arquivo = f"resultado_agregado/obs_{obs}/experimento_dimensao_{dimensao}_obs_{obs}_expoente_fuzzy_{expoente}.csv"
            df = pd.read_csv(caminho_arquivo, dtype=str)

            lista_dataframes.append(df)


df_concatenado = pd.concat(lista_dataframes)
df_concatenado.to_csv(
    "resultado_geral.csv",
    encoding="utf-8",
    index=False,
)
