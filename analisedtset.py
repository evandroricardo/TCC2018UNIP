import pandas
import etl
from sklearn.linear_model import LinearRegression


nome_arquivo = "tesouroIPCA2005.csv"

csv = pandas.read_csv(nome_arquivo, sep=";")
csv["data"] = csv["data"].map(etl.data_para_datetime)
csv["taxaCompra"] = csv["taxaCompra"].map(etl.porcentagem_para_float)
csv["taxaVenda"] = csv["taxaVenda"].map(etl.porcentagem_para_float)
csv["puCompra"] = csv["puCompra"].map(etl.real_para_float)
csv["puVenda"] = csv["puVenda"].map(etl.real_para_float)
csv = csv[["data", "taxaCompra", "taxaVenda", "puCompra", "puVenda"]]

#head
# print(csv.head(50))

lnr = LinearRegression()
lnr.fit(csv["puCompra"][:200].values.reshape(-1, 1), csv["taxaCompra"][:200].values.reshape(-1, 1))
pred = lnr.predict(csv["puCompra"].values.reshape(-1, 1))

print("ok")

#shape
#print(csv.shape)

#describe data set 
#print(csv.describe())