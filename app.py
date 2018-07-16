import pandas
from rede import Network
import etl


nome_arquivo = "tesouroIPCA2005.csv"

csv = pandas.read_csv(nome_arquivo, sep=";", )
csv["data"] = csv["data"].map(etl.data_para_datetime)
csv["taxaCompra"] = csv["taxaCompra"].map(etl.porcentagem_para_float)
csv["taxaVenda"] = csv["taxaVenda"].map(etl.porcentagem_para_float)
csv["puCompra"] = csv["puCompra"].map(etl.real_para_float)
csv["puVenda"] = csv["puVenda"].map(etl.real_para_float)


network = Network([784, 30, 10])


def dataset_para_tuplas(dataset):
    return [tuple(x) for x in dataset.values]


def amostra(tamanho: int, campos: list=None):
    if campos is not None:
        subset = csv[campos].sample(tamanho)
    else:
        subset = csv.sample(tamanho)
    return dataset_para_tuplas(subset)


def treinar(amostra, epocas, lote, eta, test):
    """
    Parâmetros de rede:
         O primeiro parametro é a amostra
         2º param é contagem de épocas
         O terceiro param é tamanho do lote
         4º param é a taxa de aprendizado (eta)
         O quinto parametro é a serie de teste
    """
    network.SGD(amostra, epocas, lote, eta, test_data=test)


def feedforward(_input: tuple):
    return network.feedforward(_input)


def evaluate(dados: list):
    return network.evaluate(dados) / len(dados)


campos = ["data", "taxaVenda"] # (x, y)
a = amostra(200, campos)
b = amostra(200, campos)
tudo = dataset_para_tuplas(csv[campos])
treinar(a, 30, 10, 3, b)           # f(x) = max(feedfoward(x))
result = evaluate(tudo)  # evaluate(x,y) = f(x) == y
print(result)
