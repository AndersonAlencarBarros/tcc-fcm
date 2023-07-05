from typing import Dict 
 
class Parametros():
    quantidade_de_bases = 3
    quantidade_de_inicializacoes = 3
    quantidade_de_agrupamentos = 2
    dimensoes = [4, 8, 16]
    quantidade_de_observacoes = [10, 30, 100]
    expoentes_fuzzy = [1.1, 2.9, 4.706, 6.512, 8.325, 15.550, 22.775, 30.0]
    
    
    @staticmethod
    def retornar_nome_da_pasta(
        quantidade_de_observacoes: int, dimensao: int, expoente: float
    ):
        self.nome_pasta: str = f"resultados/obs_{quantidade_de_observacoes}/experimento_dimensao_{dimensao}_expoente_fuzzy_{expoente}/"
        
        return nome_pasta
    
    
    @staticmethod
    def retornar_nome_do_arquivo(iteracao: int):
        nome_arquivo: str = f"{self.nome_pasta}/experimento_base_{iteracao}.csv"
        
        
    @staticmethod
    def salvar_dicionario_em_csv(resultado: Dict, nome):
        import pandas as pd
        
        df = pd.DataFrame(dicionario_hullermeier, ignore_index=True)
        df.to_csv(
            f"resultado_agregado/obs_{quantidade_de_observacoes}/experimento_dimensao_{dimensao}_expoente_fuzzy_{expoente}.csv",
            encoding="utf-8",
            index=False,
        )
        