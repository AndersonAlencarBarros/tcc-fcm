from fcm import FCM
from pprint import pprint


def _ler_json(caminho_arquivo: str):
    import json
 
    with open(caminho_arquivo) as arquivo: 
        return json.load(arquivo)


def ler_base_de_dados(dimensao: int, observacoes: int):
    caminho_arquivo = f'../bases/dimensao_{dimensao}/obs_{observacoes}.json'
    
    return _ler_json(caminho_arquivo)


def ler_inicializacao(observacoes: int, n_clusters: int):
    caminho_arquivo = f'../inicializacao/init_{observacoes}/init_{n_clusters}.json'
    
    return _ler_json(caminho_arquivo)


if __name__ == "__main__":
    dimensoes = [2, 4, 8, 16]
    observacoes = [10, 100, 1000, 10000]
    qnt_agrupamentos = [
        [2, 4, 6, 9],
        [2, 34, 66, 99],
        [2, 334, 666, 999],
        [2, 3334, 6666, 9999],
    ] 
    
    # dados = ler_inicializacao(observacoes=100, n_clusters=2)
    dados = ler_base_de_dados(dimensao=2, observacoes=10000)
    
    pprint(len(dados["bases"]))
      
    
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
