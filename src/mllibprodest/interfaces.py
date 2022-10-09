# ----------------------------------------------------------------------------------------------------
# Implementação das interfaces padrões para publicação de modelos no ML Worker.
#
# A função deste arquivo é a definição das interfaces padrões que devem ser utilizadas para fazer a
# publicação dos modelos que serão consumidos através do ML Worker.
#
# Uso: Distribua estas definições para os interessados em publicar os modelos.
#
# Fonte: https://realpython.com/python-interface/
# Baseado na seção: "Using Abstract Method Declaration"
# ----------------------------------------------------------------------------------------------------
import abc
from .shared_classes import CommonMethods
from math import ceil


class ModelPublicationInterfaceCLF(CommonMethods, metaclass=abc.ABCMeta):
    """
    Interface para publicação de modelos de ML com a função de classificação.
    Leia a documentação dos métodos e implemente-os seguindo as recomendações de tipos de parâmetros e retorno.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'predict') and
                callable(subclass.predict) and
                hasattr(subclass, 'evaluate') and
                callable(subclass.evaluate) and
                hasattr(subclass, 'get_model_info') and
                callable(subclass.get_model_info) or
                NotImplemented)

    @abc.abstractmethod
    def predict(self, dataset: list) -> list:
        """
        Faz predições utilizando o modelo que está em produção.
            :param dataset: Lista com os dados utilizados como features para realizar a predição.
            :return: Lista com os labels preditos.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, data_features: list, data_targets: list) -> dict:
        """
        Calcula as métricas para avaliação do modelo.
            :param data_features: Lista com os dados utilizados como features para realizar o cálculo.
            :param data_targets: Lista com os targets correspondentes às features para realizar o cálculo.
                                 É imprescindível que a posição de cada elemento da lista de features corresponda à
                                 resposta esperada para cada elemento da lista de targets.
            :return: Dicionário com as métricas calculadas.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_info(self) -> dict:
        """
        Obtém informações sobre o modelo que está em produção.
            :return: Dicionário com informações sobre o modelo.
        """
        raise NotImplementedError


class ModelPublicationInterfaceRETRAIN(CommonMethods, metaclass=abc.ABCMeta):
    """
    Interface para publicação de modelos de ML com a função de retreino.
    Leia a documentação dos métodos e implemente-os seguindo as recomendações de tipos de parâmetros e retorno.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_model_name') and
                callable(subclass.get_model_name) and
                hasattr(subclass, 'get_model_provider_name') and
                callable(subclass.get_model_provider_name) and
                hasattr(subclass, 'get_experiment_name') and
                callable(subclass.get_experiment_name) and
                hasattr(subclass, 'get_dataset_provider_name') and
                callable(subclass.get_dataset_provider_name) and
                hasattr(subclass, 'evaluate') and
                callable(subclass.evaluate) and
                hasattr(subclass, 'retrain') and
                callable(subclass.retrain) or
                NotImplemented)

    @abc.abstractmethod
    def get_model_name(self) -> str:
        """
        Obtém o nome do modelo que estará em produção.
            :return: Nome do modelo que estará em produção.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_provider_name(self) -> str:
        """
        Obtém o nome do provider que proverá o modelo que estará em produção.
            :return: Nome do provider que proverá o modelo que estará em produção.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_experiment_name(self) -> str:
        """
        Obtém o nome do experimento que será utilizado em produção.
            :return: Nome do experimento que será utilizado em produção.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset_provider_name(self) -> str:
        """
        Obtém o nome do provider do dataset usado pelo modelo que estará em produção.
            :return: Nome do provider do dataset usado pelo modelo que estará em produção.
        """
        raise NotImplementedError

    @staticmethod
    def generate_batch_indices(dataset_size: int, batch_size: int = 100000):
        """
        Gera indices para auxiliar na predição de registros em lote. Esta função gera uma lista contendo tuplas de
        indices (inicio, fim) que pode ser percorrida em um laço para obtenção dos índices para fazer o fatiamento do
        dataset e auxiliar a predição em lote. Ex. de lista gerada: [(0, 5), (5, 10), (10, 15), ...].
            :param dataset_size: Tamanho do dataset que será predito.
            :param batch_size: Tamanho do lote.
            :return: Lista contendo tuplas de indices (inicio, fim).
        """
        if dataset_size <= 0:
            return []

        tamanho_minimo_lote = 10000
        indices = []  # Guarda as tuplas de indices que serão utilizadas para fatiar o dataset

        if batch_size >= tamanho_minimo_lote:
            ind_inicial = 0
            ind_final = batch_size if batch_size < dataset_size else dataset_size
            qtd_lotes = ceil(dataset_size / batch_size)

            for i in range(qtd_lotes):
                indices.append((ind_inicial, ind_final))
                ind_inicial = ind_final

                # Evita que o último indice seja maior que o tamanho do dataset
                if ind_inicial + batch_size < dataset_size:
                    ind_final += batch_size
                else:
                    ind_final = dataset_size
        else:
            indices.append((0, dataset_size))

        return indices

    @abc.abstractmethod
    def evaluate(self, model, datasets: dict, baseline_metrics: dict, training_params: dict,
                 artifacts_path: str = "temp_area", batch_size: int = 100000) -> (bool, dict):
        """
        Faz a avaliação do modelo que está em produção e compara com as métricas de baseline definidas para ele.
            :param model: Modelo que está em produção.
            :param datasets: Dicionário com os datasets que serão utilizados na avaliação. Dica: Colocar o tipo de
                             dataset (features, targets, etc.) como chave e o dataset em si como valor.
            :param baseline_metrics: Dicionário com as métricas do modelo em produção que servirão de baseline para a
                                     avaliação.
            :param training_params: Dicionário com os parâmetros utilizados no treinamento do modelo que está em
                                    produção.
            :param artifacts_path: Caminho local para a obtenção dos artefatos do modelo. Para facilitar, utilize o
                                   valor padrão 'temp_area'.
            :param batch_size: Tamanho do lote. Utilizado para datasets grandes, para não faltar memória ao realizar
                               as predições.
            :return: Tupla contendo: True se o modelo passou na avaliação das métricas ou False, caso contrário, e
                                     um dicionário com informações adicionais sobre a avaliação.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def retrain(self, production_model_name: str, production_params: dict, experiment_name: str, datasets: dict,
                reasons: dict):
        """
        Faz o retreinamento do modelo de forma automatizada.
            :param production_model_name: Nome do modelo que está em produção para obtenção dos parâmetros para o
                                          retreino, se necessário.
            :param production_params: Dicionário com os parâmetros que foram utilizados no treinamento do modelo que
                                      está em produção.
            :param experiment_name: Nome do experimento para persistir o modelo retreinado.
            :param datasets: Dicionário com os datasets que serão utilizados no retreino. Dica: Colocar o tipo de
                             dataset (features, targets, etc.) como chave e o dataset em si como valor.
            :param reasons: Dicionário com o(s) motivo(s) para realização do retreinamento e/ou informações adicionais.
        """
        raise NotImplementedError
