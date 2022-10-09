# ----------------------------------------------------------------------------------------------------
# Funções úteis que poderão ser utilizadas em qualquer parte do código.
# ----------------------------------------------------------------------------------------------------
import minio
import logging
from dotenv import load_dotenv, find_dotenv


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
            logging.info(msg)

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
        logging.error(msg)
        raise RuntimeError(msg)

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
        msg = f"Não foi possível encontrar o arquivo no caminho '{file_path}'. Programa abortado!"
        logging.error(msg)
        raise FileNotFoundError(msg)
    except PermissionError:
        msg = f"Não foi possível ler o arquivo no caminho '{file_path}'. Permissão de leitura negada. Programa " \
              f"abortado!"
        logging.error(msg)
        raise PermissionError(msg)

    return bytes_arq


def validate_params(received_params: list, expected_params: dict) -> tuple:
    """
    Valida os parâmetros que foram recebidos via linha de comando na execução de um programa. Utiliza somente o '='
    como separador entre o parâmetro e seu valor.
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


def make_log(path: str):
    """
    Cria um arquivo para geração de logs. Se o mesmo já existir, inicia a gravação a partir do final desse arquivo.
    Depois de executar este método, para gravar os logs basta importar o pacote 'logging' e mandar salvar as
    mensagens de log com as funções: 'logging.error', 'logging.warning' ou 'logging.info', de acordo com o nível
    de criticidade da mensagem. Para mais opções, consulte a documentação do pacote 'logging'.
        :param path: Caminho do arquivo de logs.
    """
    # Configuração de parâmetros para geração de logs
    try:
        logging.basicConfig(filename=path,
                            format='%(asctime)s - %(levelname)s | %(funcName)s: %(message)s',
                            level=logging.INFO)
    except FileNotFoundError:
        msg = f"Não foi possível encontrar/criar o arquivo de log no caminho '{path}'. Programa abortado!"
        raise FileNotFoundError(msg)
    except PermissionError:
        msg = f"Não foi possível criar/acessar o arquivo de log no caminho '{path}'. Permissão de escrita/leitura " \
              f"negada. Programa abortado!"
        raise PermissionError(msg)
