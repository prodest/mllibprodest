# ----------------------------------------------------------------------------------------------------
# Implementação das funcionalidades da interface ModelPublicationInterfaceRETRAIN
# ----------------------------------------------------------------------------------------------------
"""
ATENÇÃO: É aconselhável gerar logs durante a execução do modelo, para facilitar o diagnóstico de problemas.

Para gravar os logs, basta salvar as mensagens de log utilizando: 'self.__logger.error', 'self.__logger.warning' ou
'self.__logger.info', conforme o nível de criticidade da mensagem. Para mais opções, consulte a documentação do
pacote 'logging'.

EXEMPLOS: self.__logger.error("Mensagem de erro"); self.__logger.info("Mensagem informativa").
"""
from .utils import *  # Para importar todas as funções/rotinas definidas no script 'utils.py'
from mllibprodest.interfaces import ModelPublicationInterfaceRETRAIN


class ModeloRETRAIN(ModelPublicationInterfaceRETRAIN):
    def __init__(self, model_name: str, model_provider_name: str, experiment_name: str, dataset_provider_name: str):
        """
        Classe para implementação das funcionalidades da interface ModelPublicationInterfaceRETRAIN.
        """
        # Cria um arquivo de log (ou utiliza o existente) e retorna um objeto logger para escrita dos logs.
        self.__logger = self.make_log(model_name + "_retrain.log")
        # Definição dos atributos necessários para a implementação dos métodos get.
        self.__model_name = model_name
        self.__model_provider_name = model_provider_name
        self.__experiment_name = experiment_name
        self.__dataset_provider_name = dataset_provider_name
        # TODO: Incluir aqui outros atributos e lógica que julgar necessário.

    # Os métodos get já foram implementados porque somente retornam valores de atributos.
    def get_model_name(self) -> str:
        return self.__model_name

    def get_model_provider_name(self) -> str:
        return self.__model_provider_name

    def get_experiment_name(self) -> str:
        return self.__experiment_name

    def get_dataset_provider_name(self) -> str:
        return self.__dataset_provider_name

    # Leia a documentação das funções disponibilizadas pelas interfaces da lib e faça uso delas! como, por exemplo:
    # 'convert_artifact_to_object', 'convert_artifact_to_pickle' e 'load_production_datasets_names'.

    # TODO: Implementar o restante dos métodos da interface ModelPublicationInterfaceRETRAIN.
