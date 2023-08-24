# ----------------------------------------------------------------------------------------------------
# Implementação das funcionalidades da interface ModelPublicationInterfaceCLF
# ----------------------------------------------------------------------------------------------------
"""
ATENÇÃO: É Aconselhável gerar logs durante a execução do modelo, para facilitar o diagnóstico de problemas.

Para gravar os logs, basta salvar as mensagens de log utilizando: 'self.__logger.error', 'self.__logger.warning' ou
'self.__logger.info', de acordo com o nível de criticidade da mensagem. Para mais opções, consulte a documentação do
pacote 'logging'.

EXEMPLOS: self.__logger.error("Mensagem de erro"); self.__logger.info("Mensagem informativa").
"""
from .utils import *  # Para importar todas as funções/rotinas definidas no script 'utils.py'
from mllibprodest.interfaces import ModelPublicationInterfaceCLF


class ModeloCLF(ModelPublicationInterfaceCLF):
    def __init__(self, model_name: str, model_provider_name: str):
        """
        Classe para implementação das funcionalidades da interface ModelPublicationInterfaceCLF.
        """
        # Cria um arquivo de log (ou utiliza o existente) e retorna um objeto logger para escrita dos logs.
        self.__logger = self.make_log(model_name + "_pub.log")
        # Atributos que serão necessários para implementar a interface.
        self.__model_name = model_name
        self.__model_provider_name = model_provider_name
        # TODO: Incluir aqui outros atributos e lógica que julgar necessário.

    # Os métodos get já foram implementados porque somente retornam valores de atributos.
    def get_model_name(self) -> str:
        return self.__model_name

    def get_model_provider_name(self) -> str:
        return self.__model_provider_name

    # Leia a documentação das funções disponibilizadas pelas inferfaces da lib e faça uso delas! como por exemplo:
    # 'load_model' e 'convert_artifact_to_object'.

    # TODO: Implementar o restante dos métodos da interface ModelPublicationInterfaceCLF.
