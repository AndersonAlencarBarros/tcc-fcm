# Análise experimental da convergência do particionamento fuzzy do método Fuzzy C-Means

_Projeto desenvolvido como Trabalho de Conclusão do Curso de Anderson de Alencar Barros_

Neste repositório de encontra todos os códigos utilizados, as bases de dados utilizadas, as inicializações, além dos resultados.

Para explorar o projeto comece fazendo um clone do projeto,

```bash
git clone https://github.com/AndersonAlencarBarros/tcc-fcm.git
```

Os requisitos para executar esse projeto são,

-   Poetry
-   Python 3.10

Ao clonar, execute para instalar todas as dependências

```bash
poetry install
```

Para executar o experimento, é preciso do Celery. O Celery pode ser iniciado da seguinte forma,

```bash
celery -A experimento worker --loglevel=INFO
```

Em outro shell, para executar qualquer código, faça

```bash
python <nome_do_arquivo.py>
```
