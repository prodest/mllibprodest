# ----------------------------------------------------------------------------------------------------
# Implementação de classes que são compartilhadas entre as interfaces padrões para publicação de
# modelos no ML Worker.
#
# Uso: Implemente aqui qualquer classe que puder ser compartilhada entre as interfaces contidas no
# arquivo 'mllibprodest/interfaces.py'
# ----------------------------------------------------------------------------------------------------
import logging
import pickle
import configparser
from typing import Union
from pathlib import Path
from .provider import Provider
from .utils import make_log


class CommonMethods:
    """
    Métodos comuns entre as interfaces dos modelos publicados.
    """
    @staticmethod
    def make_log(filename: str):
        """
        Cria um arquivo para geração de logs. Se o mesmo já existir, inicia a gravação a partir do final desse arquivo.
        Depois de executar este método, para gravar os logs basta importar o pacote 'logging' e mandar salvar as
        mensagens de log com as funções: 'logging.error', 'logging.warning' ou 'logging.info', de acordo com o nível
        de criticidade da mensagem. Para mais opções, consulte a documentação do pacote 'logging'.
            :param filename: Nome do arquivo de logs.
        """
        make_log(filename)

    @staticmethod
    def load_datasets(datasets_filenames: dict, provider: str = 'minio') -> dict:
        """
        Carrega os datasets necessários para o modelo. Os parâmetros de acesso deverão ser fornecidos através de um
        arquivo chamado '.env' que deve ser criado no repositório local e preenchido com as seguintes variáveis:
        MINIO = "nome do servidor s3", ACCESS_KEY = "chave de acesso", SECRET_KEY = "senha de acesso" e
        BUCKET = "nome do bucket", no caso do provider 'minio', ou somente a variável LOCAL_PATH = "caminho local onde
        os datasets se encontram", se o provider for 'local'. Dica de segurança: Não deixe o arquivo '.env' ser
        versionado/persistido no repositório remoto do código.
            :param datasets_filenames: Dicionário contendo os tipos de datasets e os nomes dos respectivos arquivos.
                                       Exemplo: {'features': 'nome_arquivo_features', 'targets': 'nome_arquivo_targets'}
            :param provider: Nome do provedor que fornecerá os datasets. Tipos de provider: 'minio' e 'local'.
            :return: Dicionário com os datasets carregados e prontos para serem lidos, por exemplo, através do pandas
                     com a função read_csv(), se for um arquivo csv. Obs.: As chaves do dicionário retornado serão as
                     mesmas informadas no parâmetro 'datasets_filenames' e os valores serão os datasets carregados.
        """
        return Provider.load_datasets(datasets_filenames=datasets_filenames, provider=provider)

    @staticmethod
    def load_production_params(model_name: str, provider: str = 'mlflow') -> dict:
        """
        Carrega os parâmetros que foram utilizados para treinar o modelo que está em produção.
            :param model_name: Nome do modelo que está em produção.
            :param provider: Nome do provedor que fornecerá os parâmetros do modelo que está em produção. Tipos de
                             provider: 'mlflow'.
            :return: Dicionário contendo os parâmetros carregados.
        """
        return Provider.load_production_params(model_name=model_name, provider=provider)

    @staticmethod
    def load_production_datasets_names(model_name: str, provider: str = 'mlflow') -> dict:
        """
        Carrega os nomes dos datasets que foram utilizados para treinar o modelo que está em produção.
            :param model_name: Nome do modelo que está em produção.
            :param provider: Nome do provedor que fornecerá os nomes dos datasets do modelo que está em produção.
                             Tipos de provider: 'mlflow'.
            :return: Dicionário contendo os nomes dos datasets carregados.
        """
        return Provider.load_production_datasets_names(model_name=model_name, provider=provider)

    @staticmethod
    def load_production_baseline(model_name: str, provider: str = 'mlflow') -> dict:
        """
        Carrega as métricas do modelo que está em produção que serão utilizadas como baseline para avaliação
        automatizada do modelo.
            :param model_name: Nome do modelo que está em produção.
            :param provider: Nome do provedor que fornecerá o baseline do modelo em produção. Tipos de provider:
                             'mlflow'.
            :return: Dicionário contendo as métricas de baseline.
        """
        return Provider.load_production_baseline(model_name=model_name, provider=provider)

    @staticmethod
    def load_model(model_name: str, provider: str = 'mlflow', artifacts_destination_path: str = 'temp_area'):
        """
        Carrega o modelo que está em produção e baixa os artefatos necessários.
            :param model_name: Nome do modelo que será carregado.
            :param provider: Nome do provedor que fornecerá o modelo. Tipos de provider: 'mlflow'.
            :param artifacts_destination_path: Caminho para onde os artefatos serão baixados.
            :return: Modelo carregado.
        """
        return Provider.load_model(model_name=model_name, provider=provider,
                                   artifacts_destination_path=artifacts_destination_path)

    @staticmethod
    def convert_artifact_to_pickle(model_name: str, artifact: object, file_name: str, path: str = "temp_area"):
        """
        Converte um artefato para o formato pickle, para facilitar a persistência.
            :param model_name: Nome do modelo em que o artefato será utilizado.
            :param artifact: Artefato que será convertido (str, list, dict, tuple, etc.).
            :param file_name: Nome do arquivo que será gerado (Dica: Use a extensão '.pkl').
            :param path: Caminho para gerar o artefato convertido.
        """
        caminho_artefato = str(Path(path) / model_name / file_name)

        try:
            arq = open(caminho_artefato, 'wb')
        except FileNotFoundError:
            msg = f"Não foi possível gerar o artefato '{file_name}' convertido. O caminho '{caminho_artefato}' está " \
                  f"incorreto. Programa abortado!"
            logging.error(msg)
            raise FileNotFoundError(msg)
        except PermissionError:
            msg = f"Não foi possível gerar o artefato '{file_name}' convertido no caminho '{caminho_artefato}'. " \
                  f"Permissão de escrita negada. Programa abortado!"
            logging.error(msg)
            raise PermissionError(msg)

        try:
            pickle.dump(artifact, arq)
        except TypeError as e:
            msg = f"Não foi possível gerar o artefato '{file_name}' com o Pickle (mensagem Pickle: {e}). Programa " \
                  f"abortado!"
            logging.error(msg)
            raise TypeError(msg)

        arq.close()

    @staticmethod
    def convert_artifact_to_object(model_name: str, file_name: str, path: str = "temp_area") -> \
            Union[list, tuple, dict, object]:
        """
        Converte um artefato que está no formato pickle para o objeto de origem.
            :param model_name: Nome do modelo ao qual o artefato pertence.
            :param file_name: Nome do arquivo que será lido e convertido.
            :param path: Caminho onde o arquivo a ser convertido se encontra.
            :return: Artefato convertido.
        """
        caminho_artefato = str(Path(path) / model_name / file_name)

        try:
            arq = open(caminho_artefato, 'rb')
        except FileNotFoundError:
            msg = f"Não foi possível converter o artefato '{file_name}'. O caminho '{caminho_artefato}' não foi " \
                  f"encontrado. Programa abortado!"
            logging.error(msg)
            raise FileNotFoundError(msg)
        except PermissionError:
            msg = f"Não foi possível converter o artefato '{file_name}' usando o caminho '{caminho_artefato}'. " \
                  f"Permissão de leitura negada. Programa abortado!"
            logging.error(msg)
            raise PermissionError(msg)

        try:
            objeto = pickle.load(arq)
        except pickle.UnpicklingError as e:
            msg = f"Não foi possível converter o artefato '{file_name}' com o Pickle (mensagem Pickle: {e}). " \
                  f"Programa abortado!"
            logging.error(msg)
            raise RuntimeError(msg)

        arq.close()

        return objeto

    @staticmethod
    def get_models_params() -> dict:
        """
        Obtém os parâmetros que serão utilizados para instanciar os modelos. Será buscado um arquivo com o nome
        'params.conf' contendo o nome dos modelos como uma seção [MODEL_NAME] e os parâmetros: 'experiment_name',
        'model_provider_name' e 'dataset_provider_name'.
            :return: Dicionário contendo como chave o nome do modelo e como valor outro dicionário com os parâmetros.
        """
        make_log("get_models_params.log")
        parametros_padroes = ["experiment_name", "model_provider_name", "dataset_provider_name"]
        parametros_faltantes_por_secao = {}
        faltou_parametro = False
        conf = configparser.ConfigParser()

        # Carrega os parâmetros
        try:
            nome_arq_params = conf.read("params.conf")
        except configparser.MissingSectionHeaderError:
            msg = "O arquivo com os parâmetros dos modelos ('params.conf') possui seções inválidas. Programa abortado!"
            logging.error(msg)
            raise RuntimeError(msg)
        except configparser.DuplicateOptionError as e:
            msg = f"Existem parâmetros duplicados. Programa abortado! Mensagem configParser: '{e}'."
            logging.error(msg)
            raise RuntimeError(msg)

        if "params.conf" not in nome_arq_params:
            msg = "Não foi possível encontrar o arquivo com os parâmetros dos modelos ('params.conf') ou não possui " \
                  "permissão para leitura. Programa abortado!"
            logging.error(msg)
            raise RuntimeError(msg)

        secoes = conf.sections()

        if not secoes:
            msg = "O arquivo com os parâmetros dos modelos ('params.conf') está vazio. Programa abortado!"
            logging.error(msg)
            raise RuntimeError(msg)

        # Verifica se tem parâmetros padrões faltantes
        for s in secoes:
            parametros_faltantes = []

            for p in parametros_padroes:
                if p not in conf[s]:
                    parametros_faltantes.append(p)

            if parametros_faltantes:
                faltou_parametro = True
                parametros_faltantes_por_secao[s] = parametros_faltantes

        if faltou_parametro:
            msg = f"Um ou mais parâmetros no arquivo 'params.conf' não foram encontrados: " \
                  f"{parametros_faltantes_por_secao}. Programa abortado!"
            logging.error(msg)
            raise RuntimeError(msg)

        parametros_por_modelo = {}

        # Obtém os parâmetros e gera o dicionário com os modelos e parâmetros
        for s in secoes:
            parametros_valor = {}

            for p in parametros_padroes:
                parametros_valor[p] = conf[s][p]

            parametros_por_modelo[s] = parametros_valor

        return parametros_por_modelo
