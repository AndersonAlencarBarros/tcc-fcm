import numpy as np
from fcm import FCM
import pandas as pd


def ler_json(dimensao: int, obs: int):
    return pd.read_json(
            f'inicializacao/init_{obs}/init_{dimensao}_{obs}.json',
            precise_float=True,
            orient='records',
        )

if __name__ == "__main__":
    dimensoes = [2, 4, 8, 16]
    observacoes = [10, 100, 1000, 10000]
    n_clusters = [
        [2, 3, 6, 9],
        [2, 24, 48, 99],
        [2, 249, 498, 999],
        [2, 2499, 4998, 9999],
    ]
    
    dados = ler_json(2, 10)  
    print(dados["u"][: 1])
    
    
    
    
    # for d in dimensoes:
    #     for obs in observacoes:
    #         dados = ler_json(d, obs)
            
    #         for n in n_clusters:
    #             for m in range(1, 36):
    #                 fcm = FCM(
    #                     n_clusters=n[0], 
    #                     mu=(1.1 ** m)
    #                 )
    #             for m in range(1, 36):
    #                 fcm = FCM(
    #                     n_clusters=n[1], 
    #                     mu=(1.1 ** m)
    #                 )
    #             for m in range(1, 36):
    #                 fcm = FCM(
    #                     n_clusters=n[2], 
    #                     mu=(1.1 ** m)
    #                 )
    #             for m in range(1, 36):
    #                 fcm = FCM(
    #                     n_clusters=n[3], 
    #                     mu=(1.1 ** m)
    #                 )

