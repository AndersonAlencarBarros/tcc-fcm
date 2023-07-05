import pandas as pd
from mpmath import *
from parametros import Parametros


p = Parametros()
    

"""
Script para plotar os resultados
"""

resultado = {
    "10 observações": {
        "4 dimensões": [],
        "8 dimensões": [],
        "16 dimensões": []
    }, 
    "30 observações": {
        "4 dimensões": [],
        "8 dimensões": [],
        "16 dimensões": []
    }, 
    "100 observações": {
        "4 dimensões": [],
        "8 dimensões": [],
        "16 dimensões": []
    }, 
}

resultado_por_dimensao = {
    "4 dimensões": {
        "10 observações": [],
        "30 observações": [],
        "100 observações": [],
    },
    "8 dimensões": {
        "10 observações": [],
        "30 observações": [],
        "100 observações": [],
    },
    "16 dimensões": {
        "10 observações": [],
        "30 observações": [],
        "100 observações": [],
    },
}

"""
Preenche os dicionários
"""

for obs in p.quantidade_de_observacoes:
    for dimensao in p.dimensoes:
        for expoente in p.expoentes_fuzzy:
            nome_arquivo: str = f"resultado_agregado/obs_{obs}/experimento_dimensao_{dimensao}_expoente_fuzzy_{expoente}.csv"
            df = pd.read_csv(nome_arquivo, dtype=str)
            
            media_de_hullermeier = mpf(df["Média do Indice de Hullermeier"][0]) 
            
            resultado[f'{obs} observações'][f"{dimensao} dimensões"].append(media_de_hullermeier)
            resultado_por_dimensao[f"{dimensao} dimensões"][f'{obs} observações'].append(media_de_hullermeier)

"""                
Exibe os Resultados
"""
                
import matplotlib.pyplot as plt
import seaborn as sns

# Criar gráficos para cada dimensão ou por quantidade de observacoes

for dimensao, observacoes in resultado.items():
    plt.figure(figsize=(8, 6))
    plt.title(f"{dimensao}")
    plt.xlabel("Expoente Fuzzy")
    plt.ylabel("Índice de Hullermeier")

    for observacao, valores in observacoes.items():
        plt.plot(p.expoentes_fuzzy, valores, label=observacao)

    plt.legend()
    plt.grid()
    plt.show()


"""
Gráfico Geral, com todas as visualizações.
"""


# # Obtém as chaves do dicionário
# obs_keys = list(resultado.keys())

# print(obs_keys)

# # Define as dimensões para plotagem
# dimensoes = ['4 dimensões', '8 dimensões', '16 dimensões']

# # Configura a precisão dos números
# mp.dps = 25

# for i, obs_key in enumerate(obs_keys):
#     obs_data = resultado[obs_key]
    
#     for j, dim_key in enumerate(dimensoes):
#         dim_data = obs_data[dim_key]
        
#         y_values = [float(nstr(value)) for value in dim_data]
        
#         # Cria um novo subplot para cada combinação de obs e dimensão
#         plt.subplot(3, 3, i * 3 + j + 1)
        
#         sns.set_style("whitegrid")
#         sns.scatterplot(x=p.expoentes_fuzzy, y=y_values, )
#         sns.lineplot(x=p.expoentes_fuzzy, y=y_values,)
        
#         plt.title(f'{obs_key}, {dim_key}')
#         plt.xlabel('Expoente Fuzzy')
#         plt.ylabel('Hullemeier')
        
#         plt.ylim(0.45, 1.05)
#         plt.xlim(1.1, 30)
        
#         plt.xticks(fontsize=10)
#         plt.yticks(fontsize=10)
#         plt.grid(True, linestyle='--', alpha=0.5)


# plt.tight_layout()
# plt.show() 
