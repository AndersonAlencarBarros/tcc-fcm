from tasks import treinamento

"""Para 10 observações"""
# observacoes = 10
# qnt_agrupamentos = [2, 4, 6, 9]

"""Para 100 observações"""
observacoes = 100
qnt_agrupamentos = [ # 66,
    99,
]  # FALTA AS DIMENSOES 4, 8 E 16 PARA 2, 34 QUANTIDADES DE CLUSTERS

"""Para 1000 observações"""
# observacoes = 1000
# qnt_agrupamentos = [2, 334, 666, 999]

dimensao = [4, 8, 16]   # 2, 
mu = [1.1, 8.325, 15.550, 22.775, 30.0]

list_tasks = []
for d in dimensao:
    for n_clusters in qnt_agrupamentos:
        list_tasks.extend(
            treinamento.s(
                dimensao=d,
                observacoes=observacoes,
                n_clusters=n_clusters,
                mu=m,
            )
            for m in mu
        )

from celery import group

group_tasks = group(list_tasks)
result = group_tasks.apply_async()
