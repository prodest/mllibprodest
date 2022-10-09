# ----------------------------------------------------------------------------------------------------
# Implementação das funcionalidades da interface ModelPublicationInterfaceRETRAIN
# ----------------------------------------------------------------------------------------------------
"""
Para gravar os logs basta salvar as mensagens de log com as funções: 'logging.error', 'logging.warning' ou
'logging.info', de acordo com o nível de criticidade da mensagem. Para mais opções, consulte a documentação do
pacote 'logging'. Ex. logging.error("Mensagem de erro"); logging.info("Mensagem informativa").
"""
import logging  # Para gerar os logs (aconselhamos gerar os logs para facilitar o diagnóstico de problemas)
from mllibprodest.interfaces import ModelPublicationInterfaceRETRAIN


class ModeloRETRAIN(ModelPublicationInterfaceRETRAIN):
    def __init__(self):  # Não receba parâmetros através do init.
        """
        Classe para implementação das funcionalidades da interface ModelPublicationInterfaceRETRAIN.
        """
        # Cria um arquivo de log ou utiliza o existente.
        self.make_log("log_retrain.log")
        # Definição dos atributos necessários para a implementação dos métodos get. Alterar de 'None' para o valor
        # correto dos atributos.
        self.__model_name = None
        self.__model_provider_name = None
        self.__experiment_name = None
        self.__dataset_provider_name = None
        # TODO: Incluir aqui outros atributos e lógica que julgar necessário.

    # Os métodos get já foram implementados porque somente retornam valores de atributos.
    # Lembre-se: altere os valores dos atributos, de 'None' para o valor correto, para retorná-los.
    def get_model_name(self) -> str:
        return self.__model_name

    def get_model_provider_name(self) -> str:
        return self.__model_provider_name

    def get_experiment_name(self) -> str:
        return self.__experiment_name

    def get_dataset_provider_name(self) -> str:
        return self.__dataset_provider_name

    # Leia a documentação das funções disponibilizadas pelas inferfaces da lib e faça uso delas! como por exemplo:
    # 'convert_artifact_to_object', 'convert_artifact_to_pickle' e 'load_production_datasets_names'.

    # TODO: Implementar o restante dos métodos da interface ModelPublicationInterfaceRETRAIN.
