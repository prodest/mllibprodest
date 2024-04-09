# ----------------------------------------------------------------------------------------------------
# Funções úteis que poderão ser utilizadas em qualquer parte do código.
# ----------------------------------------------------------------------------------------------------
import minio
import logging
from logging.handlers import RotatingFileHandler
import configparser
from pathlib import Path
from os import makedirs
from dotenv import load_dotenv, find_dotenv
from os import environ as env


def make_log(filename: str) -> logging.Logger:
    """
    Cria um logger para gerar logs na console ou gravar em um arquivo. Se o arquivo já existir, inicia a gravação a
    partir do final dele.
        :param filename: Nome do arquivo de logs (caso o log seja gravado em arquivo).
        :return: Um logger para geração dos logs.
    """
    # Para controlar a gravação de logs em arquivo ou não
    stack_log_output = env.get('STACK_LOG_OUTPUT')

    if not stack_log_output:
        stack_log_output = "file"

    # Configurações básicas
    logger_name = filename.split(".")[0]
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s | %(funcName)s: %(message)s')

    if stack_log_output == "console":
        logger.propagate = False
        consolehandler = logging.StreamHandler()
        consolehandler.setLevel(logging.INFO)
        consolehandler.setFormatter(formatter)
        logger.addHandler(consolehandler)
    elif stack_log_output == "file":
        # Se a pasta de logs não existir, cria
        try:
            makedirs("logs", exist_ok=True)
        except PermissionError:
            msg = "Erro ao criar a pasta 'logs'. Permissão de escrita negada!"
            raise PermissionError(msg) from None

        log_file_path = str(Path("logs") / filename)

        # Configuração de parâmetros para gravação de logs
        try:
            rotatehandler = RotatingFileHandler(log_file_path, mode='a', maxBytes=10485760, backupCount=5)
            rotatehandler.setLevel(logging.INFO)
            rotatehandler.setFormatter(formatter)
            logger.addHandler(rotatehandler)
        except FileNotFoundError:
            msg = f"Não foi possível encontrar/criar o arquivo de log no caminho '{log_file_path}'."
            raise FileNotFoundError(msg) from None
        except PermissionError:
            msg = f"Não foi possível criar/acessar o arquivo de log no caminho '{log_file_path}'. Permissão de " \
                f"escrita/leitura negada."
            raise PermissionError(msg) from None
    else:
        raise ValueError(f"A variável de ambiente 'STACK_LOG_OUTPUT' contém um tipo de saída do log incorreto "
                         f"('{stack_log_output}'). Os possíveis valores são: 'console' ou 'file'.")

    return logger


# Para facilitar, define um logger único para todas as funções
LOGGER = make_log("LOG_MLLIB.log")


def load_env_variables(path: str = ""):
    """
    Carrega as variáveis de ambiente armazenadas no arquivo '.env'.
        :param path: Caminho para o arquivo '.env'. Caso não seja especificado, fará uma busca nas pastas filhas da
                     pasta onde o programa está localizado para encontrar o arquivo '.env'.
    """
    if type(path) is str:
        if path == "":
            dotenv_path = find_dotenv(usecwd=True)
        else:
            dotenv_path = path

        if dotenv_path != "":
            msg = f"Utilizando as configurações de ambiente obtidas através do arquivo: '{dotenv_path}'"
            LOGGER.info(msg)

        load_dotenv(dotenv_path)


def get_file_s3(file_name: str, s3_server: str, access_key: str, secret_key: str, bucket: str):
    """
    Obtém um arquivo através do protocolo s3.
        :param file_name: Nome do arquivo a ser baixado.
        :param s3_server: Servidor s3 que proverá o arquivo.
        :param access_key: Chave de acesso para logar no servidor s3.
        :param secret_key: Senha para logar no servidor s3.
        :param bucket: Nome do bucket onde o arquivo a ser baixado se encontra.
        :return: Objeto contendo o arquivo baixado.
    """
    client = minio.Minio(s3_server, access_key, secret_key)

    try:
        obj_arquivo = client.get_object(bucket_name=bucket, object_name=file_name)
    except minio.error.S3Error as e:
        msg = f"Não foi possível obter o arquivo desejado. Mensagem do servidor S3: {e}"
        LOGGER.error(msg)
        raise RuntimeError(msg) from None

    return obj_arquivo.read()


def get_file_local(file_path: str):
    """
    Obtém um arquivo através do armazenamento local.
        :param file_path: Caminho do arquivo a ser carregado.
        :return: Objeto contendo o arquivo carregado.
    """
    try:
        with open(file_path, 'rb') as arq:
            bytes_arq = arq.read()
    except FileNotFoundError:
        msg = f"Não foi possível encontrar o arquivo no caminho '{file_path}'."
        LOGGER.error(msg)
        raise FileNotFoundError(msg) from None
    except PermissionError:
        msg = f"Não foi possível ler o arquivo no caminho '{file_path}'. Permissão de leitura negada."
        LOGGER.error(msg)
        raise PermissionError(msg) from None

    return bytes_arq


def validate_params(received_params: list, expected_params: dict) -> tuple:
    """
    Valida os parâmetros recebidos via linha de comando na execução de um programa. Utiliza somente o '=' como
    separador entre o parâmetro e seu valor.
        :param received_params: Parâmetros recebidos via linha de comando no momento da execução de um programa.
                                Obs.: A lista de parâmetros deve ser obtida através da propriedade 'sys.argv'.
        :param expected_params: Nomes de parâmetros e tipos esperados pelo programa que servirão como base para a
                                validação. Ex.: {'nome': str, 'idade': int}.
        :return: (True, dict). Verdadeiro, se passou na validação, e um dicionário com os nomes dos parâmetros esperados
                 e seus respectivos valores ou (False, str). Falso, se não passou na validação, e uma string com os
                 erros encontrados.
    """
    if type(expected_params) is dict:
        if len(expected_params) == 0:
            return False, "O dicionário (expected_params) contendo os parâmetros esperados está vazio."
    else:
        return False, "O tipo do parâmetro 'expected_params' não é dict."

    # Não conta o primeiro parâmetro porque é o nome do script que foi executado
    if len(received_params) <= 1:
        return False, "A lista de parâmetros recebidos (received_params) está vazia."

    parametros_validados = {}
    erros_encontrados = ""

    for p in received_params[1:]:
        nome_valor = p.split("=")

        if len(nome_valor) == 2:
            nome_param = nome_valor[0]
            valor_param = nome_valor[1]

            if nome_param in expected_params:
                try:
                    # Tenta converter para o tipo de parâmetro esperado
                    parametros_validados[nome_param] = expected_params[nome_param](valor_param)
                except ValueError:
                    erros_encontrados += f"O valor para o parâmetro '{nome_param}' deve ser do tipo " \
                                         f"{expected_params[nome_param]}.\n"
                except TypeError:
                    erros_encontrados += f"O tipo do parâmetro '{nome_param}' informado no dicionário " \
                                         f"'expected_params' está incorreto. Não digite o tipo entre aspas."
            else:
                erros_encontrados += f"O parâmetro '{nome_param}' não existe.\n"
        else:
            erros_encontrados += f"Não foi possível encontrar o separador de valores '=' no parâmetro '{p}' " \
                                 f"ou existem espaços ' = ' entre o nome do parâmetro, separador e valor.\n"

    if erros_encontrados == "":
        return True, parametros_validados
    else:
        return False, erros_encontrados


def get_models_params(path: str = "") -> dict:
    """
    Obtém os parâmetros que serão utilizados para instanciar os modelos. Será buscado um arquivo com o nome
    'params.conf' contendo o nome dos modelos como uma seção [MODEL_NAME] e os parâmetros: 'source_file',
    'model_class', 'experiment_name', 'model_provider_name' e 'dataset_provider_name'.
        :param path: Caminho onde se encontra o arquivo de parâmetros. O padrão é estar na pasta local.
        :return: Dicionário contendo como chave o nome do modelo e como valor outro dicionário com os parâmetros.
    """
    parametros_padroes = ["source_file", "model_class", "experiment_name", "model_provider_name",
                          "dataset_provider_name"]
    parametros_faltantes_por_secao = {}
    faltou_parametro = False
    conf = configparser.ConfigParser()
    param_file_path = str(Path(path) / "params.conf")

    # Carrega os parâmetros
    try:
        nome_arq_params = conf.read(param_file_path)
    except configparser.MissingSectionHeaderError as e:
        msg = f"O arquivo com os parâmetros dos modelos ('params.conf') possui seções inválidas. Mensagem " \
              f"configParser: '{e}'."
        LOGGER.error(msg)
        raise ValueError(msg) from None
    except configparser.DuplicateSectionError as e:
        msg = f"Existem seções duplicadas. Mensagem configParser: '{e}'."
        LOGGER.error(msg)
        raise ValueError(msg) from None
    except configparser.DuplicateOptionError as e:
        msg = f"Existem parâmetros duplicados. Mensagem configParser: '{e}'."
        LOGGER.error(msg)
        raise ValueError(msg) from None

    if "params.conf" not in nome_arq_params:
        msg = "Não foi possível encontrar o arquivo com os parâmetros dos modelos ('params.conf') ou não possui " \
              "permissão para leitura."
        LOGGER.error(msg)
        raise RuntimeError(msg)

    secoes = conf.sections()

    if not secoes:
        msg = "O arquivo com os parâmetros dos modelos ('params.conf') está vazio."
        LOGGER.error(msg)
        raise ValueError(msg)

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
        msg = f"Um ou mais parâmetros não foram encontrados no arquivo 'params.conf'. Parâmetros faltantes por " \
              f"modelo: {parametros_faltantes_por_secao}."
        LOGGER.error(msg)
        raise ValueError(msg)

    parametros_por_modelo = {}

    # Obtém os parâmetros e gera o dicionário com os modelos e parâmetros
    for s in secoes:
        parametros_valor = {}

        for p in parametros_padroes:
            parametros_valor[p] = conf[s][p]

        parametros_por_modelo[s] = parametros_valor

    return parametros_por_modelo
