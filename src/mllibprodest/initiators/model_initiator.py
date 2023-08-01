# ---------------------------------------------------------------------------------------------------------
# Classes e funções para inicialização de modelos de ML (Machine Learning).
# ---------------------------------------------------------------------------------------------------------
import logging
import importlib
from ..utils import get_models_params


class InitModels:
    """
    Classe utilizada para instanciar os modelos de ML (Machine Learning).
    """
    @staticmethod
    def init_models(path: str = "") -> dict:
        """
        Inicia os modelos de ML (Machine Learning) utilizando os parâmetros configurados no arquivo 'params.conf'.
            :param path: Caminho onde se encontra o arquivo de parâmetros. O padrão é estar na pasta local.
            :return: Dicionário contendo como chave os nomes dos modelos e como valor os modelos instanciados.
        """
        modelos = {}
        models_params = get_models_params(path)

        for model_name in models_params.keys():
            caminho_import = f"models.{models_params[model_name]['source_file']}"

            try:
                modulo = importlib.import_module(caminho_import, package=None)
            except ModuleNotFoundError:
                msg = f"O módulo '{caminho_import}' não foi encontrado. Verifique no arquivo 'params.conf' se o " \
                      f"parâmetro 'source_file' foi informado corretamente e/ou se este módulo está dentro da pasta " \
                      f"'models'. Programa abortado!"
                logging.error(msg)
                raise ModuleNotFoundError(msg) from None

            if models_params[model_name]['model_class'] == "ModeloCLF":
                cls = getattr(modulo, "ModeloCLF")

                if callable(cls) and type(cls).__name__ == 'ABCMeta':
                    try:
                        modelos[model_name] = cls(model_name=model_name,
                                                  model_provider_name=models_params[model_name]['model_provider_name'])
                    except TypeError as e:
                        msg = f"Faltou a implementação do(s) seguinte(s) método(s): {str(e)[65:]}."
                        logging.error(msg)
                        raise TypeError(msg) from None
                else:
                    msg = f"O tipo do 'ModeloCLF' está incorreto: '{type(cls).__name__}'. 'ModeloCLF' deve ser uma " \
                          f"classe que herda os métodos da interface 'ModelPublicationInterfaceCLF' e possua as " \
                          f"implementações para os métodos abstratos dela."
                    logging.error(msg)
                    raise TypeError(msg)
            elif models_params[model_name]['model_class'] == "ModeloRETRAIN":
                cls = getattr(modulo, "ModeloRETRAIN")

                if callable(cls) and type(cls).__name__ == 'ABCMeta':
                    try:
                        modelos[model_name] = cls(model_name=model_name,
                                                  model_provider_name=models_params[model_name]['model_provider_name'],
                                                  experiment_name=models_params[model_name]['experiment_name'],
                                                  dataset_provider_name=models_params[model_name]['dataset_provider_name'])
                    except TypeError as e:
                        msg = f"Faltou a implementação do(s) seguinte(s) método(s): {str(e)[65:]}."
                        logging.error(msg)
                        raise TypeError(msg) from None
                else:
                    msg = f"O tipo do 'ModeloRETRAIN' está incorreto: '{type(cls).__name__}'. 'ModeloRETRAIN' deve " \
                          f"ser uma classe que herda os métodos da interface 'ModelPublicationInterfaceRETRAIN' e " \
                          f"possua as implementações para os métodos abstratos dela."
                    logging.error(msg)
                    raise TypeError(msg)
            else:
                msg = f"O valor do parâmetro 'model_class' está incorreto. Foi informado " \
                      f"'{models_params[model_name]['model_class']}' no arquivo 'params.conf', porém deve ser " \
                      f"'ModeloCLF' ou 'ModeloRETRAIN'. Programa abortado!"
                logging.error(msg)
                raise ValueError(msg)

        return modelos
