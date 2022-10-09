# ----------------------------------------------------------------------------------------------------
# Implementação das funcionalidades da interface ModelPublicationInterfaceCLF
# ----------------------------------------------------------------------------------------------------
"""
Para gravar os logs basta salvar as mensagens de log com as funções: 'logging.error', 'logging.warning' ou
'logging.info', de acordo com o nível de criticidade da mensagem. Para mais opções, consulte a documentação do
pacote 'logging'. Ex. logging.error("Mensagem de erro"); logging.info("Mensagem informativa").
"""
import logging  # Para gerar os logs (aconselhamos gerar os logs para facilitar o diagnóstico de problemas)
from mllibprodest.interfaces import ModelPublicationInterfaceCLF


class ModeloCLF(ModelPublicationInterfaceCLF):
    def __init__(self):  # Não receba parâmetros através do init.
        """
        Classe para implementação das funcionalidades da interface ModelPublicationInterfaceCLF.
        """
        # Cria um arquivo de log ou utiliza o existente.
        self.make_log("log_pub.log")
        # Atributos que serão necessários para implementar a interface. Altere os seus valores se precisar.
        self.__model_name = "model_name"
        self.__provider = "mlflow"
        # TODO: Incluir aqui outros atributos e lógica que julgar necessário.

    # Leia a documentação das funções disponibilizadas pelas inferfaces da lib e faça uso delas! como por exemplo:
    # 'load_model' e 'convert_artifact_to_object'.

    # TODO: Implementar o restante dos métodos da interface ModelPublicationInterfaceCLF.
