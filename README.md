# Biblioteca de ML (*Machine Learning*) - Prodest 
A finalidade desta biblioteca é prover interfaces e funções que dão suporte ao provisionamento de modelos de ML na Stack 
de ML do Prodest. 

*Workflow* básico para construção, disponibilização e publicação de modelos:

![](https://github.com/prodest/mllibprodest/blob/main/docs/workflow.png?raw=true)

## Pré-requisitos
* **Python >= 3.8.** Instruções: [Linux (Geralmente já vem instalado por padrão)](https://python.org.br/instalacao-linux) ou [Windows](https://www.python.org/downloads/windows).
* **Git.** Instruções: [Linux](https://git-scm.com/download/linux) ou [Windows](https://git-scm.com/download/win).
* **Venv.** Gerenciador de ambiente virtual Python adotado no tutorial. Instruções: [Linux e Windows (escolha o sistema na página)](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment). 
Ou qualquer outro gerenciador de ambiente Python que preferir.

## 1. Realize experimentos e escolha o modelo 

Esta é uma das etapas iniciais de um projeto para o desenvolvimento de um modelo de *Machine Learning*. Neste momento é 
necessário entender o problema a ser resolvido; levantar requisitos; obter e tratar os dados, etc. Também é nessa etapa 
que se verifica a viabilidade (ou não) da construção de um modelo.

Neste passo você tem **total liberdade** para construir o seu modelo e realizar os experimentos que quiser. Entretanto, 
é importante que os resultados e artefatos gerados pelos experimentos, desde já, sejam registrados para facilitar a 
comparação dos resultados obtidos e a publicação do modelo. Esta lib utiliza o 
[MLflow](https://github.com/mlflow/mlflow) como plataforma para registro dos experimentos/modelos (no contexto da lib, o 
MLflow é chamado de *Provider*).

Apesar do registro dos experimentos ser importante, deixar de registrá-los agora **não** vai impedir que você construa o 
seu modelo!

Você tem duas opções: 
- Seguir com a construção do modelo e execução dos experimentos e, caso chegue à conclusão de que o modelo é viável, 
ajustar o código para realizar o registro; ou


- Fazer uma pausa e entender primeiro como registrar seus experimentos no MLflow e já construir o código com a lógica
necessária para isso.

Independente da opção escolhida, haverá necessidade de, agora ou depois, aprender (caso não saiba) como registrar os 
experimentos do modelo no MLflow. 
Para alcançar esse objetivo, leia a [documentação oficial do MLflow](https://mlflow.org/docs/latest/index.html).

Segue abaixo, um exemplo simples de como registrar os experimentos de um modelo construído com o [scikit-learn](https://scikit-learn.org) 
utilizando o MLflow.

```python
import os
import mlflow.sklearn  # Importa o sklearn através do MLflow
import pickle  # Para gerar um artefato de exemplo

# Obs.: Utilize as duas linhas abaixo, exatamente como apresentadas, para configurar o parâmetro 'Tracking URI' do MLflow 
# nos seus códigos de testes. Dessa forma, quando subir para produção não haverá necessidade de modificá-las, pois lá
# o parâmetro 'Tracking URI' será obtido diretamente através da variável de ambiente 'MLFLOW_TRACKING_URI'.
if os.environ.get('MLFLOW_TRACKING_URI') is None:
    mlflow.set_tracking_uri('sqlite:///teste_mlflow.db')

# Configura o experimento (se não existir, cria)
mlflow.set_experiment(experiment_name="Teste sklearn")

# Inicia uma execução do experimento (um experimento pode possuir várias execuções)
mlflow.start_run(run_name="t1", description="teste 1")

# Registra algumas informações adicionais no experimento (coloque as informações que julgar necessárias, no formato dict)
tags = {"Projeto": "Teste", "team": "ML", "util": "Informação útil"}
mlflow.set_tags(tags)

# Inicia o registro dos logs da execução do sklearn
mlflow.sklearn.autolog()

# TODO: Inclua aqui a lógica para fazer o fit do modelo

'''Exemplo de modelo, somente para o propósito de testes!'''
# Adaptado de https://scikit-learn.org/stable/modules/tree.html#classification
import matplotlib
import numpy as np
from sklearn import tree
X = np.array([[0, 0], [1, 1]])
Y = np.array([0, 1]).reshape(-1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
'''Fim do exemplo de modelo.'''

'''
Salva um artefato de seu interesse no MLflow (podem ser arquivos em diversos formatos: txt, pkl, png, jpeg, etc.).
Exemplos de artefato: gráficos, objetos persistidos com pickle, enfim, tudo que for relevante e/ou necessário para
que o modelo funcione e/ou para análise das execuções.
'''
# Cria um aterfato de teste no formato pickle (obs.: todas as classes da lib tem os métodos 
# 'convert_artifact_to_pickle' e 'convert_artifact_to_object' para auxiliar na persistência dos artefatos)
artefato = {"t": 1}
caminho_artefato = "artefato.pkl"
with open(caminho_artefato, 'wb') as arq:
    pickle.dump(artefato, arq)

# Salva o artefato criado
mlflow.log_artifact(caminho_artefato)

# Finaliza o experimento
mlflow.end_run()

print("\nTeste finalizado!\n")
```

Se você quiser testar um registro de experimento através do código acima, faça o seguinte:

- Crie uma pasta para testes;
- Copie e cole o código acima em um editor de texto simples e salve com o nome 'testeml.py' dentro da pasta criada;
- Abra um prompt de comando ou terminal e entre na pasta criada;
- Crie e ative um ambiente virtual Python, conforme instruções: [Linux e Windows (escolha o sistema na página)](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment);
- Instale os pacotes mlflow, sklearn, matplotlib e numpy;

```bash
pip install mlflow sklearn matplotlib numpy
```
- Rode o teste (ignore as mensagens do tipo 'INFO' de criação do banco de dados);
```bash
python testeml.py
```
Cabe observar que: depois de rodar o código de teste, foi criada uma pasta chamada '**mlruns**', dentro da pasta de 
testes, que serve para armazenar os artefatos gerados pelo código e que são apresentados na interface do MLflow. 
Abaixo segue uma listagem do conteúdo gerado pelo código de teste (obs.: essa parte do caminho vai ser diferente de 
acordo com cada experimento/execução realizados: '1/f454220e01bd43e7b66c2354c20f0085'. O conteúdo da pasta também será 
diferente de acordo com cada modelo).
```bash
(env) user:~/testes/mlruns/1/f454220e01bd43e7b66c2354c20f0085/artifacts$ dir
artefato.pkl  model  training_confusion_matrix.png  training_precision_recall_curve.png  training_roc_curve.png
```
Dentro da pasta criada para testes também foi gerado um arquivo chamado '**teste_mlflow.db**', que é um pequeno banco 
de dados [SQlite](https://www.sqlite.org), que serve para armazenar os modelos que foram registrados.

- Inicie o servidor do MLflow;

Perceba que a pasta '**mlruns**' e o arquivo '**teste_mlflow.db**' foram passados como parâmetros na hora de iniciar o 
servidor, para que o experimento de teste possa ser visualizado. Portanto, é **mandatório** sempre iniciar o servidor do
MLflow **de dentro da pasta** onde se encontra o código que fará o registro dos artefatos e dos experimentos/modelos.

**DICA**: Abra um outro prompt de comando ou terminal diferente; entre na pasta onde se encontra o código para registro 
dos experimentos/modelo; execute o comando para iniciar o servidor do MLflow de dentro desta pasta. Pois assim, você 
conseguirá rodar o código e já observar os resultados sem ter que parar o servidor para liberar o prompt ou terminal.
```bash
mlflow server --backend-store-uri sqlite:///teste_mlflow.db --host 0.0.0.0 -p 5000 --default-artifact-root mlruns
```

- Verifique se o experimento foi registrado. Acesse o MLFlow: [http://localhost:5000](http://localhost:5000) e procure 
pelo experimento/execução '**Teste sklearn**' na seção **Experiments** (se o experimento não estiver listado, verifique 
se o servidor foi iniciado de dentro da pasta correta);


- Clique na execução do experimento que se encontra na coluna '**Created**' (destaque em verde);

![](https://github.com/prodest/mllibprodest/blob/main/docs/experiments-mlflow.png?raw=true)
- Verifique se os artefatos foram gravados;

![](https://github.com/prodest/mllibprodest/blob/main/docs/artifacts-mlflow.png?raw=true)

- Finalize o servidor do MLflow. Faça 'CTRL+c' no prompt de comando ou terminal onde ele foi iniciado;
- Apague a pasta criada para realização dos testes.

**NOTA**: Existem vários outros *frameworks* suportados: TensorFlow, Keras, Pytorch, etc. (veja a lista completa para 
Python em [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)), inclusive é possível registrar 
modelos que **não são suportados nativamente** pelo MLflow utilizando a função 
[mlflow.pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html).

**ATENÇÃO**: Sua interação direta com o MLflow será somente para registro dos experimentos/modelo. Essa interação é 
essencial porque dá liberdade ao desenvolvedor para escolher o *framework* que achar mais adequado para construção
do seu modelo. A lib disponibiliza funções para obtenção do modelo registrado e dos seus artefatos, além de 
outras funções relacionadas à carga de *datasets*. Leia a documentação das interfaces, classes e funções da lib para 
mais detalhes.

### Antes de ir para os próximos passos...
Quando você já tiver realizado vários experimentos utilizando o MLflow e decidido por colocar o modelo em produção, 
será preciso registrar o modelo treinado para que o mesmo seja carregado e usado na construção dos *workers*, conforme 
descrito no passo 3. Siga as instruções abaixo para registrar o modelo:

- Caso o servidor do MLflow não esteja rodando, entre na pasta onde o **script que salvará o experimento** (código 
desenvolvido para criação do modelo) se encontra; 
- Ative o ambiente virtual Python. Instruções: [Linux e Windows (escolha o sistema na página)](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment); 
- Inicie o servidor do MLflow;
```bash
mlflow server --backend-store-uri sqlite:///teste_mlflow.db --host 0.0.0.0 -p 5000 --default-artifact-root mlruns
```
- Acesse o MLflow ([http://localhost:5000](http://localhost:5000)) e clique no experimento que foi criado por você (se 
o experimento não estiver listado, verifique se o servidor do MLflow foi iniciado de dentro da pasta correta);
- Clique no link (que está na coluna **'Created'**) para a rodada do experimento que deseja registrar;
- Clique no botão **'Register Model'** e escolha a opção **'Create New Model'**;
- Dê um nome para o modelo e clique em **'Register'**;
- Na barra superior clique em **'Models'**;
- Clique no link para a última versão do modelo que está em **'Latest Version'**;
- Na opção **'Stage'** troque de **'None'** para **'Production'** e clique em OK.

Quando for testar a implementação dos *workers* (passo 3), lembre de deixar o servidor do MLflow rodando para que seja 
possível carregar o modelo.

## 2. Organize o código de acordo com o *template*
Uma vez que o modelo foi desenvolvido e testado, agora é o momento de iniciar as tratativas para publicá-lo na *stack* de ML do 
Prodest. Porém, antes, é oporturno mostrar como o modelo será integrado à *stack*. Esta integração se dará através de 
componentes denominados *workers*, cuja codificação é de responsabilidade de quem está construindo o modelo. Na 
ilustração abaixo é possível observar que os *workers* são acessados pelos componentes de apoio da *stack* para 
permitir a publicação dos modelos. Caso seja necessário, uma mesma stack poderá publicar mais de um modelo, desde que 
sejam construídos *workers* dedicados para cada um deles.

![](https://github.com/prodest/mllibprodest/blob/main/docs/stack-ml.png?raw=true)

Existem dois tipos de *workers*:
- **worker_pub**: Fornece os métodos necessários para publicação do modelo.
- **worker_retrain**: Responsável pela avaliação do desempenho do modelo e retreinamento, se for preciso.

Para que o modelo possa ser publicado, é imprescindível que a organização do código seja conforme especificado na pasta
'**templates**' (esta pasta vem junto com repositório da lib). 

![](https://github.com/prodest/mllibprodest/blob/main/docs/estrutura-pastas.png?raw=true)

As regras são simples mas precisam ser seguidas, caso contrário a publicação do modelo falhará.
- Os nomes das pastas '**worker_pub**' e '**worker_retrain**' não podem ser alterados;
- Os nomes dos scripts padrões contidos nestas pastas não podem ser alterados;
- (Opcional, mas recomendável). Separe as funções utilitárias para o funcionamento dos *workers* nos arquivos '**utils.py**';
- Gere um arquivo de *requirements* para cada um dos *workers* **separadamente**. Dica: Use um ambiente virtual Python 
separado para cada *worker*, instale os pacotes requeridos para o funcionamento deles e no final gere um arquivo 
'**requirements.txt**' para cada *worker*;
- Não importe código de fora destas pastas. Se os dois *workers* precisarem de uma mesma função, faça uma cópia desta em 
cada pasta (o arquivo 'utils.py' pode ajudar a organizar estas funções!);
- Cuide para que os importes funcionem corretamente, dentro de cada pasta, **sem precisar** configurar a variável de 
ambiente PYTHONPATH;
- Utilize a pasta '**temp_area**' para salvar e ler os arquivos temporários que forem criados.

**NOTA**: Os scripts '**mytest_pub.py**' e '**mytest_retrain.py**' podem ser utilizados por você para criação de testes
personalizados, para isso basta implementar a função '**test**' em cada um deles. Já os scripts '**test_pub.py**' e 
'**test_retrain.py**' podem ser usados para testar se algumas premissas foram atendidas, através de testes padrões 
da lib e a execução automática dos testes personalizados que foram implementados pelo usuário. No passo 3 é mostrado 
como rodar os scripts '**test_pub.py**' e '**test_retrain.py**'.

Caso queira, você pode criar pastas ou arquivos de apoio dentro das pastas dos *workers* para organizar seu código, 
desde que não modique a localização dos arquivos especificada pelos *templates*.

Para obter e utilizar a pasta com os *templates*:
- Clone o repositório da lib;
```bash
git clone https://github.com/prodest/mllibprodest.git
```

- Entre na pasta gerada no processo de clonagem do repositório e copie o conteúdo da pasta '**templates**' para outro local 
de sua preferência (não trabalhe na pasta do repositório).

## 3. Implemente as interfaces da biblioteca

Antes de iniciar a implementação das interfaces, é importante criar um ambiente virtual Python **separadamente** para cada 
*worker*. Dessa forma você conseguirá gerar os arquivos de *requirements* sem maiores problemas. Siga as instruções abaixo:

**Para o worker_pub**:
- Abra um prompt de comando ou terminal;
- Entre na pasta para onde você copiou o conteúdo da pasta '**templates**'; 
- Entre na pasta '**worker_pub**', crie e ative um ambiente virtual Python, conforme instruções: [Linux e Windows (escolha o sistema na página)](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment); 
- Instale a lib para o worker_pub;
```bash
pip install mllibprodest
```
- Feche o prompt de comando ou terminal.

**Para o worker_retrain**:

- Abra **outro** prompt de comando ou terminal (**Não** aproveite o anterior de forma alguma, pois dará errado!);
- Entre na pasta para onde você copiou o conteúdo da pasta '**templates**'; 
- Entre na pasta '**worker_retrain**', crie e ative **outro** ambiente virtual Python;
- Instale a lib para o worker_retrain;
```bash
pip install mllibprodest
```
- Feche o prompt de comando ou terminal.

Pronto. Agora você tem um ambiente virtual Python para cada *worker*. Quando for utilizar uma IDE ou editor de código 
para implementar as interfaces, configure para que eles utilizem os ambientes virtuais criados para os respectivos 
*workers*. Dessa forma, à medida que você for produzindo o código e necessitar de instalar pacotes, esses serão 
instalados nos ambientes virtuais criados. Quando terminar a implementação, basta você gerar os arquivos de 
*requirements* com base no ambiente virtual de cada *worker* separadamente. Acredite, isso vai te ajudar bastante!

Outro ponto importante antes de implementar as interfaces é saber que: para publicar o modelo será necessário a criação 
de três artefatos obrigatórios, inclusive seguindo o mesmo nome (*case sensitive*). Estes artefatos devem ser 
dicionários (dict) salvos com o [Pickle](https://docs.python.org/3/library/pickle.html) (utilize a função 
'convert_artifact_to_pickle' quando estiver implementando as intefaces):

- **TrainingParams.pkl**: Deve conter os parâmetros que você escolheu utilizar no treinamento do modelo. Não há 
necessidade de colocar os parâmetros nos quais você manteve os valores *default*. Você pode colocar outros parâmetros 
criados por você necessários para que o modelo funcione. Coloque o nome 
do parâmetro como nome da chave e o valor do parâmetro como valor da chave. Ex. baseado no *DecisionTreeClassifier*: 
{'criterion': 'entropy', 'max_depth': '20', 'random_state': '77', 'meu_parametro_personalizado': 'teste'}. 


- **TrainingDatasetsNames.pkl**: Deve conter os tipos de datasets e os nomes dos respectivos arquivos utilizados no
treinamento do modelo. Exemplo: {'features': 'nome_arquivo_features', 'targets': 'nome_arquivo_targets'}.


- **BaselineMetrics.pkl**: Deve conter as métricas que você achar relevantes para decidir se o modelo precisa ser 
retreinado. Por exemplo, você poderia definir a métrica acurácia mínima e caso o modelo que estiver em produção, ao ser 
avaliado, não estiver atingindo o valor dessa métrica, será um indicativo de que ele precisa ser retreinado. Outro exemplo 
claro da necessidade de retreinamento é quando um modelo de classificação é treinado para predizer um conjunto de *labels* 
e por um motivo qualquer surgem novos *labels*, nesse caso o modelo não saberá predizer estes *labels* e necessitará 
ser retreinado em um dataset atualizado com os novos *labels*. Exemplo: {'acuracia_minima': 0.94, 
'labels_presentes_no_treino': ['gato', 'cachorro']}.

**NOTA**: Estes artefatos deverão ser criados pelo script utilizado para registro dos experimentos no processo de 
treinamento do modelo e salvos através da função '**mlflow.log_artifact**', no momento da realização dos experimentos. Os 
artefatos salvos junto com o modelo devem ser utilizados na implementação das funcionalidades das interfaces no momento 
da construção dos *workers*. A única maneira de obter parâmetros e informações acerca do modelo registrado será por 
intermédio destes artefatos. Por favor, não persista nada localmente, pois os *workers* não trocarão mensagens nem 
compartilharão acesso à dados entre si.

Para implementar as interfaces e construir os *workers* basta editar os *templates* conforme abaixo:

**REGRAS**: Implemente todos os métodos solicitados respeitando os tipos dos parâmetros e de retorno. Não troque os 
nomes dos parâmetros.

**worker_pub**:

- Abra o arquivo '**pub.py**', que se encontra na pasta '**worker_pub**', e implemente os métodos da interface 
**ModelPublicationInterfaceCLF** através da classe **ModeloCLF**. Leia os comentários, eles te guiarão na implementação.

**worker_retrain**:

- Abra o arquivo '**retrain.py**', que se encontra na pasta '**worker_retrain**', e implemente os métodos da interface 
**ModelPublicationInterfaceRETRAIN** através da classe **ModeloRETRAIN**. Leia os comentários, eles te guiarão na 
implementação.

A lib disponibiliza vários métodos úteis que auxiliarão na implementação das interfaces. 
Todos os métodos estão documentados via [docstrings](https://peps.python.org/pep-0257/) que, geralmente, são 
renderizadas pelas IDEs ou editores de código facilitando a leitura da documentação. Veja alguns métodos úteis disponíveis:

- **make_log** - Criação do arquivo para geração de logs.
- **load_datasets** - Carga de datasets.
- **load_model** - Carga de modelos salvos.
- **load_production_params**, **load_production_datasets_names**, **load_production_baseline** - Carga das informações 
dos modelos publicados, salvas através dos artefatos obrigatórios.
- **convert_artifact_to_pickle** - Conversão de um artefato para o formato pickle.
- **convert_artifact_to_object** - Conversão de um artefato que está no formato pickle para o objeto de origem.

Explore a documentação para saber das possibilidades de uso da lib.

### Teste o código produzido!
O repositório da lib disponibiliza os scripts '**test_pub.py**' e '**test_retrain.py**' para realização de testes para 
verificar se alguns requisitos solicitados estão sendo atendidos. Também é possível criar testes personalizados através da 
implementação da função '**test**' que se encontra nos scripts '**mytest_pub.py**' e '**mytest_retrain.py**'. Todos 
estes scripts estão nas pastas **worker_pub** e **worker_retrain**.

Para testar o seu código siga as instruções abaixo:

- Caso o servidor do MLflow não esteja rodando, entre na pasta onde o código/script para registro dos experimentos/modelo 
se encontra; ative o ambiente virtual Python, instruções: [Linux e Windows (escolha o sistema na página)](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment),
e inicie o servidor do MLflow:
```bash
mlflow server --backend-store-uri sqlite:///teste_mlflow.db --host 0.0.0.0 -p 5000 --default-artifact-root mlruns
```
- Obtenha o caminho completo da pasta '**mlruns**' (ela é criada dentro da pasta onde o script para geração dos 
experimentos/modelo foi executado);


- Se for testar o *worker* pub, entre na pasta '**worker_pub**' e execute o comando abaixo. Lembre-se de informar o 
caminho completo da pasta '**mlruns**' através do parâmetro '**--mlruns_path**';
```bash
python test_pub.py --mlruns_path="caminho completo para a pasta mlruns"
```
- Se for testar o *worker* retrain, entre na pasta '**worker_retrain**' e execute o comando abaixo. Lembre-se de 
informar o caminho completo da pasta '**mlruns**' através do parâmetro '**--mlruns_path**';
```bash
python test_retrain.py --mlruns_path="caminho completo para a pasta mlruns"
```

Leia atentamente as mensagens e caso exista alguma inconsistência no teste, atenda ao que for solicitado pelo script.

## 4. Disponibilize o código para publicação do modelo

Antes de enviar os códigos, certifique-se que eles estão funcionando de acordo com as regras estabelecidas e que os 
arquivos com os *requirements* foram gerados corretamente. Se ocorrer algum erro que impeça a publicação, entraremos 
em contato para informar o ocorrido e fornecer as informações sobre o erro. 

Para disponibilizar o modelo para publicação:

- Crie uma pasta chamada '**publicar**';
- Copie as pastas '**worker_pub**' e '**worker_retrain**' para a pasta '**publicar**' (**não** incluir a pasta 
'**env**', que é do ambiente virtual Python, nem a pasta '**temp_area**', que é utilizada para guardar arquivos 
temporários) ;
- Gere um arquivo de *requirements.txt* para o código responsável pelo registro dos experimentos/modelo;
- Copie a pasta que contém o código responsável pelo registro dos experimentos/modelo para a pasta publicar (**não** 
incluir a pasta '**env**' e arquivos desnecessários);
- Compacte a pasta '**publicar**' utilizando o formato '.zip';
- Envie o arquivo '**publicar.zip**' para o Prodest, conforme alinhamento prévio realizado em reunião ou qualquer 
outro meio de contato.

