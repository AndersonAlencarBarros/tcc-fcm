def _ler_json(caminho_arquivo: str):
    import json

    with open(caminho_arquivo) as arquivo:
        return json.load(arquivo)


def ler_base_de_dados(dimensao: int, observacoes: int):
    import numpy as np

    caminho_arquivo = f"../bases/dimensao_{dimensao}/obs_{observacoes}.json"

    dados = _ler_json(caminho_arquivo)
    return np.array(dados["bases"])


def ler_inicializacao(iteracao: int, observacoes: int, n_clusters: int):
    import numpy as np

    caminho_arquivo = (
        f"../inicializacao/init_{observacoes}_{n_clusters}/init{iteracao}.json"
    )

    dados = _ler_json(caminho_arquivo)
    return np.array(dados["u"])
