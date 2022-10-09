# ----------------------------------------------------------------------------------------------------
# Provider para obtenção de datasets persistidos na área de armazenamento local
# ----------------------------------------------------------------------------------------------------
import os
import logging
from io import BytesIO
from ..utils import load_env_variables, get_file_local
from pathlib import Path


def load_datasets_local(datasets_filenames: dict) -> dict:
    """
    Carrega os datasets necessários para o modelo que foram persistidos na área de armazenamento local. Os parâmetros
    de acesso deverão ser fornecidos através de um arquivo chamado '.env' que deve ser criado no repositório local e
    preenchido com a seguinte variável: LOCAL_PATH = "caminho local onde os datasets se encontram". Dica de
    segurança: Não deixe o arquivo '.env' ser versionado/persistido no repositório remoto do código.
        :param datasets_filenames: Dicionário contendo os tipos de datasets e os nomes dos respectivos arquivos.
                                   Exemplo: {'features': 'nome_arquivo_features', 'targets': 'nome_arquivo_targets'}
        :return: Dicionário com os datasets carregados.
    """
    # Obtém as informações necessárias para carregar os arquivos
    load_env_variables()
    local_path = os.environ.get("LOCAL_PATH")

    if local_path is None:
        msg = f"Não foram encontradas todas as variáveis de ambiente necessárias. Certifique-se que um arquivo " \
              f"chamado '.env' exista; esteja localizado na pasta da aplicação e que possua valor para a variável: " \
              f"'LOCAL_PATH'. Ou se preferir, configure essa variável de ambiente e seu respectivo valor. " \
              f"Programa abortado!"
        logging.error(msg)
        raise RuntimeError(msg)

    datasets = {}

    for tipo, nome_arquivo in datasets_filenames.items():
        datasets[tipo] = BytesIO(get_file_local(str(Path(local_path) / nome_arquivo)))

    return datasets
