from tasks import treinamento

"""Para 10 observações"""
# observacoes = 10
# dimensao = [2, 4, 8, 16]
# qnt_agrupamentos = [2, 4, 6, 9]
# mu = [1.1, 8.325, 15.550, 22.775, 30.0]

"""Para 100 observações"""
observacoes = 100
dimensao = [2, 4, 8, 16]
qnt_agrupamentos = [2, 34, 66, 99]
mu = [1.1, 8.325, 15.550, 22.775, 30.0]

"""Para 1000 observações"""
# observacoes = 1000
# dimensao = [2, 4, 8, 16]
# qnt_agrupamentos = [2, 334, 666, 999]
# mu = [1.1, 8.325, 15.550, 22.775, 30.0]

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
