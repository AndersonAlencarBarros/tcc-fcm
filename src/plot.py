import numpy as np
import pandas as pd
from mpmath import *
import math
  
  
quantidade_de_bases = 3
quantidade_de_observacoes = [10, 30, 100]
dimensoes = [4, 8, 16]
expoentes_fuzzy = [1.1, 8.325, 15.550, 22.775, 30.0]


"""
Lê os resultados e escolhe o Hullermeier de menor custo

Plota os resultados
"""

resultado = {
    "obs_10": {
        "dimensao_4": [],
        "dimensao_8": [],
        "dimensao_16": []
    }, 
    "obs_30": {
        "dimensao_4": [],
        "dimensao_8": [],
        "dimensao_16": []
    }, 
    "obs_100": {
        "dimensao_4": [],
        "dimensao_8": [],
        "dimensao_16": []
    }, 
}

for obs in quantidade_de_observacoes:
    for dimensao in dimensoes:
        for expoente in expoentes_fuzzy:
            nome_pasta: str = f"resultados/obs_{obs}/experimento_dimensao_{dimensao}_obs_{obs}_expoente_fuzzy_{expoente}/"

            menor_custo = mpf(math.inf)
            hullermeier_menor_custo = mpf(0)
            for i in range(quantidade_de_bases):
                nome_arquivo: str = f"{nome_pasta}/experimento_inicializacao_{i}.csv"
                df = pd.read_csv(nome_arquivo, dtype=str)
                                
                custo_atual = mpf(df["custo"][0])                               
                hullermeier_atual = mpf(df["hullermeier"][0]) 
                if custo_atual < menor_custo:
                    menor_custo = custo_atual
                    hullermeier_menor_custo = hullermeier_atual
            
            resultado[f'obs_{obs}'][f"dimensao_{dimensao}"].append(hullermeier_menor_custo)
                

from pprint import pprint
pprint(resultado)

import matplotlib.pyplot as plt
import seaborn as sns

# Define os valores de x
x_values = [1.1, 8.325, 15.55, 22.775, 30.0]

# Obtém as chaves do dicionário
obs_keys = list(resultado.keys())

# Define as dimensões para plotagem
dimensoes = ['dimensao_4', 'dimensao_8', 'dimensao_16']

# Configura a precisão dos números
mp.dps = 25

# Loop pelos valores de observação (obs_10, obs_30, obs_100)
for i, obs_key in enumerate(obs_keys):
    obs_data = resultado[obs_key]
    
    # Loop pelas dimensões (dimensao_4, dimensao_8, dimensao_16)
    for j, dim_key in enumerate(dimensoes):
        dim_data = obs_data[dim_key]
        
        # Converte os valores mpmath em float para plotagem
        y_values = [float(nstr(value)) for value in dim_data]
        
        # Cria um novo subplot para cada combinação de obs e dimensão
        plt.subplot(3, 3, i * 3 + j + 1)
        
        sns.set_style("whitegrid")
        sns.scatterplot(x=x_values, y=y_values, )
        sns.lineplot(x=x_values, y=y_values,)
        
        plt.title(f'{obs_key}, {dim_key}')
        plt.xlabel('Expoente Fuzzy')
        plt.ylabel('Hullemeier')
        
        plt.ylim(0, 1.1)
        plt.xlim(1.1, 30)
        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show() 