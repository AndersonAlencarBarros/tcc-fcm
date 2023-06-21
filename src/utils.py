def _ler_json(caminho_arquivo: str):
    import json

    with open(caminho_arquivo) as arquivo:
        return json.load(arquivo)


def ler_base_de_dados(iteracao: int, dimensao: int, observacoes: int):
    import numpy as np

    caminho_arquivo = f"bases/dimensao_{dimensao}_obs_{observacoes}/base_{iteracao}.json"

    dados = _ler_json(caminho_arquivo)
    return np.array(dados["base"])


def ler_inicializacao(iteracao: int, observacoes: int, n_clusters: int):
    import numpy as np

    caminho_arquivo = f"inicializacao/init_{observacoes}/init{iteracao}.json"

    dados = _ler_json(caminho_arquivo)
    return np.array(dados["u"])

def retorna_particao_base(quantidade_de_agrupamentos: int, quantidade_de_observacoes: int):
    from mpmath import fdiv
    import numpy as np
    
    return np.full((quantidade_de_agrupamentos, quantidade_de_observacoes), fdiv(1, quantidade_de_agrupamentos))