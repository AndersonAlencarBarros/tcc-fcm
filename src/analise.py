import numpy as np
import pandas as pd
from mpmath import *
from hullermeier import hullermeier
from fcm import __verificar_soma_igual_a_1__


mp.dps = 400


def retornar_matrix_pertinencia(string, n_clusters: int, observacoes: int):
    import re

    pattern = r"\d+\.\d+"
    numeros = re.findall(pattern, string)
    numeros = [mpf(num) for num in numeros]

    return np.reshape(np.array(numeros), (n_clusters, observacoes))

observacoes = 10
qnt_agrupamentos = [2, 4, 6, 9]
dimensao = [2, 4, 8, 16]
mu = [1.1, 8.325, 15.550, 22.775, 30.0]


resultado = {
    "dimensao_2": {
        "2": [],
        "4": [],
        "6": [],
        "9": [],
    },
    "dimensao_4": {
        "2": [],
        "4": [],
        "6": [],
        "9": [],
    },
    "dimensao_8": {
        "2": [],
        "4": [],
        "6": [],
        "9": [],
    },
    "dimensao_16": {
        "2": [],
        "4": [],
        "6": [],
        "9": [],
    },
}


# for d in dimensao:
#     for n_clusters in qnt_agrupamentos:
#         for m in mu:
#             caminho_arquivo = f"../resultados/experimento_10/experimento_obs_{observacoes}_dim_{d}_nc_{n_clusters}_mu_{m}.csv"

#             df = pd.read_csv(caminho_arquivo, index_col=False, dtype=str)

#             u_infinito = retorna_particao_base(n_clusters, observacoes)

#             print(u_calculado.shape)
#             __verificar_soma_igual_a_1__(u_calculado)

#             h = hullermeier(u_calculado, u_infinito)

#             resultado[f'dimensao_{d}'][f'{n_clusters}'].append(h)

# from pprint import pprint
# pprint(resultado)

# import matplotlib.pyplot as plt

# for i, dimensao in enumerate(resultado.keys()):
#     plt.subplot(2, 2, i+1)  # Configurar subplot
#     plt.title(dimensao)     # Configurar título do gráfico

#     for tamanho, dados in resultado[dimensao].items():
#         plt.plot(mu, dados, label=f'Q. Clusters {tamanho}')  # Plotar linha para cada tamanho

#     plt.legend()     # Adicionar legenda
#     plt.xlabel('m')  # Configurar label do eixo x
#     plt.ylabel('Hullermeier')  # Configurar label do eixo y

# plt.tight_layout()  # Ajustar layout dos subplots
# plt.show()  # Exibir os gráficos dos subplots
# plt.show()  # Exibir os gráficos


# base = Path(f"resultado/resultado_{observacoes}")
# jsonpath = base / "resultado.json"

# base.mkdir(exist_ok=True)
# jsonpath.write_text(dumps(resultado))

if __name__ == "__main__":
    u_calculado = retornar_matrix_pertinencia(df["u"][0], n_clusters=2, observacoes=10)

    print(u_calculado)
