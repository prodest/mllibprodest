# Publicando a Biblioteca
Para empacotar a biblioteca de ML do Prodest e publicá-la no [Python Package Index (PyPI)](https://pypi.org/) siga os passos abaixo.

Este passo a passo foi baseado no tutorial [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/). Se desejar obter maiores detalhes, consulte o tutorial completo.

## 1. Clone o repositório
Abra um prompt de comando ou terminal e rode o comando abaixo.
```bash
git clone https://github.com/prodest/mllibprodest.git
```

## 2. Crie um ambiente virtual Python e ative-o
Entre na pasta criada no processo de clonagem do repositório.
```bash
cd mllibprodest
```
Crie o ambiente virtual.
```bash
python3.10 -m venv env
```
Ative o ambiente virtual.
```bash
source env/bin/activate
```

## 3. Se necessário, atualize os dados do projeto
Para fazer alguma correção e/ou melhoria no pacote que já está publicado, é obrigatório o incremento da versão. 
Portanto é preciso editar o arquivo *'pyproject.toml'* e alterar a variável *'version'*, conforme exemplo abaixo. No caso, essa
variável poderia ser alterada para '1.8.3', se a mudança for pequena, ou 1.9.0 se a mudança for um pouco maior. Essa decisão 
deverá levar em conta a política de versionamento adotada.

Outra configuração comum de ser alterada são as dependências da biblioteca. Se alguma dependência for atualizada, 
a nova versão deverá ser informada na variável *'dependencies'*. 


```toml
[project]
...
version = "1.8.2"
...
dependencies = ['minio==7.2.1', 'python-dotenv==1.0.0', 'mlflow==2.9.2', 'boto3==1.34.7']
```


## 4. Faça o *'build'* da biblioteca
Antes de rodar os comandos para construir o pacote com a biblioteca, atualize o *pip*, *build* e *twine*. 

```bash
python -m pip install --upgrade pip build twine
```

De dentro da pasta criada no processo de clonagem do repositório (mllibprodest), rode os comandos abaixo.

- Constrói o pacote e salva na pasta *'dist'*.
```bash
python -m build
```

- Envia o pacote para o PyPI. Caso existam mais pacotes nesta pasta, é necessário especificar quais deles
serão enviados.

Obs.: Será solicitado o fornecimento de usuário e senha; ou 'token' de acesso cadastrados no site [PyPI](https://pypi.org/).
```bash
twine upload dist/*
```

Se este processo ocorrer com sucesso, a biblioteca, ou nova versão dela, estará publicada no PyPI podendo ser instalada 
através do comando *'pip install mllibprodest'*.
