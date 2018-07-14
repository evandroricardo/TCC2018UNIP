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


network = Network([2, 3, 1])
# network.update_mini_batch([(0.5, 0.7)], 0.005)


def dataset_para_tuplas(dataset):
    return [tuple(x) for x in dataset.values]


def amostra(tamanho: int, campos: list=None):
    if campos is not None:
        subset = csv[campos].sample(tamanho)
    else:
        subset = csv.sample(tamanho)
    return dataset_para_tuplas(subset)


def treinar(amostra, tamanho):
    network.SGD(amostra, len(amostra), tamanho, 0.005)


def feedforward(_input: tuple):
    return network.feedforward(_input)

def evaluate(dados: list):
    return network.evaluate(dados) / len(dados)

campos = ["puCompra", "puVenda"] # (x, y)
a = amostra(200, campos)
tudo = dataset_para_tuplas(csv[campos])
treinar(a, 50)           # f(x) = max(feedfoward(x))
result = evaluate(tudo)  # evaluate(x,y) = f(x) == y
print(result)
