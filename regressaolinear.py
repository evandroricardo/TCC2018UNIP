from sklearn.linear_model import LinearRegression
from dados import csv

#head
# print(csv.head(50))

lnr = LinearRegression()
lnr.fit(csv["puCompra"][:200].values.reshape(-1, 1), csv["taxaCompra"][:200].values.reshape(-1, 1))
pred = lnr.predict(csv["puCompra"].values.reshape(-1, 1))

print(pred[:10])
print("ok")

#shape
#print(csv.shape)

#describe data set 
#print(csv.describe())