import numpy as np
import pandas as pd
from mpmath import *
from parametros import Parametros
import os


p = Parametros()
mp.dps = 400
  

"""
Lê os 3 resultados para cada inicialização e faz a média do índice de Hullermeier
"""


for quantidade_de_observacoes in p.quantidade_de_observacoes:
    for dimensao in p.dimensoes:
        for expoente in p.expoentes_fuzzy:
            nome_pasta: str = (
                f"resultados/obs_{quantidade_de_observacoes}/"
                f"experimento_dimensao_{dimensao}_expoente_fuzzy_{expoente}/"
            )
            
            media_hullermeier = mpf(0)
            dict_hullermeier = {"Hullermeier 0": 0, "Hullermeier 1": 0, "Hullermeier 2": 0}
            for i in range(p.quantidade_de_bases):
                nome_arquivo: str = f"experimento_base_{i}.csv"
                df = pd.read_csv(f"{nome_pasta}/{nome_arquivo}", dtype=str)


                dict_hullermeier[f"Hullermeier {i}"] = df["hullermeier"][0]
                media_hullermeier += mpf(df["hullermeier"][0])
 
            dicionario_media_hullermeier = {
                "quantidade de observações": quantidade_de_observacoes,
                "dimensão": dimensao,
                "expoente fuzzy": expoente,
                "Hullermeier 0": dict_hullermeier["Hullermeier 0"],
                "Hullermeier 1": dict_hullermeier["Hullermeier 1"],
                "Hullermeier 2": dict_hullermeier["Hullermeier 2"],
                "Média do Indice de Hullermeier": media_hullermeier / p.quantidade_de_bases,
            }

            nome_pasta: str = f"resultado_agregado/obs_{quantidade_de_observacoes}"
            if not os.path.exists(nome_pasta):
                os.makedirs(nome_pasta) 

            df = pd.DataFrame([dicionario_media_hullermeier])
            nome_arquivo: str = (
                f"experimento_dimensao_{dimensao}"
                f"_expoente_fuzzy_{expoente}.csv"
            )
            df.to_csv(
                f"{nome_pasta}/{nome_arquivo}",
                encoding="utf-8",
                index=False,
            )



if __name__ == "__main__":
    """
    Gera o Resultado Agregado

    Lê todos os resultados e concatena em um único csv.
    """
    

    lista_dataframes = []
    for obs in p.quantidade_de_observacoes:
        for dimensao in p.dimensoes:
            for expoente in p.expoentes_fuzzy:
                caminho_arquivo = (
                    f"resultado_agregado/obs_{obs}/"
                    f"experimento_dimensao_{dimensao}"
                    f"_expoente_fuzzy_{expoente}.csv"
                )
                df = pd.read_csv(caminho_arquivo, dtype=str)

                lista_dataframes.append(df)


    df_concatenado = pd.concat(lista_dataframes)
    df_concatenado.to_csv(
        "resultado_geral.csv",
        encoding="utf-8",
        index=False,
    )