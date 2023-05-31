import pandas as pd
import os

pasta = "experimento_10"

dataframes = []
for arquivo in os.listdir(pasta):
    if arquivo.endswith(".csv"):
        caminho_arquivo = os.path.join(pasta, arquivo)
        df = pd.read_csv(caminho_arquivo)
        dataframes.append(df)

df_completo = pd.concat(dataframes)

grouped = df_completo.groupby(["dimensão", "quantidade de agrupamentos", "mu"])[
    "tempo de execução (s)"
].mean()

print(grouped)
