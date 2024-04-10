# ---------------------------------------------------------------------------------------------------------
# Classes e funções para inicialização de modelos de ML (Machine Learning).
# ---------------------------------------------------------------------------------------------------------
import importlib
from ..utils import get_models_params, make_log

# Para facilitar, define um logger único para todas as funções
LOGGER = make_log("LOG_MLLIB.log")


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
            except ModuleNotFoundError as e:
                msg_erro = str(e)

                if caminho_import in msg_erro:
                    msg = f"Modelo: {model_name}. O módulo '{caminho_import}' não foi encontrado. Verifique no " \
                          f"arquivo 'params.conf' se o parâmetro 'source_file' foi informado corretamente e/ou se " \
                          f"este módulo está dentro da pasta 'models'."
                    LOGGER.error(msg)
                    raise ModuleNotFoundError(msg) from None
                else:
                    msg = f"Modelo: {model_name}. Erro ao importar os módulos necessários para o módulo " \
                          f"'{caminho_import}'. Mensagem do Import: {msg_erro}"
                    LOGGER.error(msg)
                    # Não utilizei o 'from None' para preservar o traceback
                    raise ModuleNotFoundError(msg)
            except ImportError as e:
                msg = f"Modelo: {model_name}. Erro ao importar os módulos necessários para o módulo " \
                      f"'{caminho_import}'. Mensagem do Import: {str(e)}"
                LOGGER.error(msg)
                # Não utilizei o 'from None' para preservar o traceback
                raise ImportError(msg)

            if models_params[model_name]['model_class'] == "ModeloCLF":
                cls = getattr(modulo, "ModeloCLF")

                if callable(cls) and type(cls).__name__ == 'ABCMeta':
                    try:
                        modelos[model_name] = cls(model_name=model_name,
                                                  model_provider_name=models_params[model_name]['model_provider_name'])
                    except TypeError as e:
                        msg = f"Faltou a implementação do(s) seguinte(s) método(s) para o modelo '{model_name}' " \
                              f"(classe 'ModeloCLF'): {str(e)[64:]}."
                        LOGGER.error(msg)
                        raise TypeError(msg) from None
                else:
                    msg = f"Modelo: {model_name}. O tipo do 'ModeloCLF' está incorreto: '{type(cls).__name__}'. " \
                          f"'ModeloCLF' deve ser uma classe que herda os métodos da interface " \
                          f"'ModelPublicationInterfaceCLF' e possua as implementações para os métodos abstratos dela."
                    LOGGER.error(msg)
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
                        msg = f"Faltou a implementação do(s) seguinte(s) método(s) para o modelo '{model_name}' " \
                              f"(classe 'ModeloRETRAIN'): {str(e)[68:]}."
                        LOGGER.error(msg)
                        raise TypeError(msg) from None
                else:
                    msg = f"Modelo: {model_name}. O tipo do 'ModeloRETRAIN' está incorreto: '{type(cls).__name__}'. " \
                          f"'ModeloRETRAIN' deve ser uma classe que herda os métodos da interface " \
                          f"'ModelPublicationInterfaceRETRAIN' e possua as implementações para os métodos " \
                          f"abstratos dela."
                    LOGGER.error(msg)
                    raise TypeError(msg)
            else:
                msg = f"Modelo: {model_name}. O valor do parâmetro 'model_class' está incorreto. Foi informado " \
                      f"'{models_params[model_name]['model_class']}' no arquivo 'params.conf', porém deve ser " \
                      f"'ModeloCLF' ou 'ModeloRETRAIN'."
                LOGGER.error(msg)
                raise ValueError(msg)

        return modelos
