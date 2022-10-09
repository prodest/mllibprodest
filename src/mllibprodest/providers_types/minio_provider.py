# ----------------------------------------------------------------------------------------------------
# Provider para obtenção de datasets persistidos no Minio
# ----------------------------------------------------------------------------------------------------
import os
import logging
from io import BytesIO
from ..utils import load_env_variables, get_file_s3


def load_datasets_minio(datasets_filenames: dict) -> dict:
    """
    Carrega os datasets necessários para o modelo que foram persistidos utilizando o Minio. Os parâmetros de acesso
    deverão ser fornecidos através de um arquivo chamado '.env' que deve ser criado no repositório local e preenchido
    com as seguintes variáveis: MINIO = "nome do servidor s3", ACCESS_KEY = "chave de acesso", SECRET_KEY = "senha de
    acesso" e BUCKET = "nome do bucket". Dica de segurança: Não deixe o arquivo '.env' ser versionado/persistido no
    repositório remoto do código.
        :param datasets_filenames: Dicionário contendo os tipos de datasets e os nomes dos respectivos arquivos.
                                   Exemplo: {'features': 'nome_arquivo_features', 'targets': 'nome_arquivo_targets'}
        :return: Dicionário com os datasets carregados.
    """
    # Obtém as credenciais e as informações necessárias para baixar os arquivos
    load_env_variables()
    s3_server = os.environ.get("MINIO")
    access_key = os.environ.get("ACCESS_KEY")
    secret_key = os.environ.get("SECRET_KEY")
    bucket = os.environ.get("BUCKET")

    if s3_server is None or access_key is None or secret_key is None or bucket is None:
        msg = f"Não foram encontradas todas as variáveis de ambiente necessárias. Certifique-se que um arquivo " \
              f"chamado '.env' exista; esteja localizado na pasta da aplicação e que possua valores para as " \
              f"variáveis: 'MINIO', 'ACCESS_KEY', 'SECRET_KEY' e 'BUCKET'. Ou se preferir, configure essas variáveis " \
              f"de ambiente e seus respectivos valores. Programa abortado!"
        logging.error(msg)
        raise RuntimeError(msg)

    datasets = {}

    for tipo, nome_arquivo in datasets_filenames.items():
        datasets[tipo] = BytesIO(get_file_s3(nome_arquivo, s3_server, access_key, secret_key, bucket))

    return datasets
