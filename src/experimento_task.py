from tasks import treinamento

quantidade_de_bases = 3
quantidade_de_observacoes = 100  # observacoes = [10, 30, 100]
quantidade_de_agrupamentos = 2
dimensoes = [4, 8, 16]
expoentes_fuzzy = [1.1, 8.325, 15.550, 22.775, 30.0]

list_tasks = []
for dimensao in dimensoes:
    for i in range(quantidade_de_bases):
        list_tasks.extend(
            treinamento.s(
                iteracao=i,
                dimensao=dimensao,
                quantidade_de_observacoes=quantidade_de_observacoes,
                quantidade_de_agrupamentos=quantidade_de_agrupamentos,
                expoente_fuzzy=expoente,
            )
            for expoente in expoentes_fuzzy
        )

from celery import group

group_tasks = group(list_tasks)
result = group_tasks.apply_async()
