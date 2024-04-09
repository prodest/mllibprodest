# ----------------------------------------------------------------------------------------------------
# Funções úteis que poderão ser utilizadas pelos providers
# ----------------------------------------------------------------------------------------------------
from ..provider import Provider
from ..utils import get_models_params


def get_models_versions_providers(path: str = "") -> dict:
    """
    Obtém as versões dos modelos que estão sendo providas pelos providers. Obs.: Esta versão é gerenciada pelo provider.
        :param path: Caminho onde se encontra o arquivo de parâmetros. O padrão é estar na pasta local.
        :return: Dicionário com o nome de cada modelo como chave e a respectiva versão como valor.
    """
    models_params = get_models_params(path)
    models_per_provider = {}
    models_version = {}

    # Obtém os nomes dos modelos e organiza por provider
    for model_name in models_params.keys():
        model_provider_name = models_params[model_name]['model_provider_name']

        if model_provider_name not in models_per_provider:
            models_per_provider[model_provider_name] = [model_name]
        else:
            models_per_provider[model_provider_name].append(model_name)

    # Busca as versões dos modelos nos respectivos providers
    for model_provider_name in models_per_provider.keys():
        models_names = models_per_provider[model_provider_name]
        models_version.update(Provider.get_models_versions(models_names=models_names, provider=model_provider_name))

    return models_version
