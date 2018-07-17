import pandas
import etl


__nome_arquivo = "tesouroIPCA2005.csv"

csv = pandas.read_csv("./data/" + __nome_arquivo, sep=";")
csv["data"] = csv["data"].map(etl.data_para_datetime)
csv["taxaCompra"] = csv["taxaCompra"].map(etl.porcentagem_para_float)
csv["taxaVenda"] = csv["taxaVenda"].map(etl.porcentagem_para_float)
csv["puCompra"] = csv["puCompra"].map(etl.real_para_float)
csv["puVenda"] = csv["puVenda"].map(etl.real_para_float)
csv = csv[["data", "taxaCompra", "taxaVenda", "puCompra", "puVenda"]]
