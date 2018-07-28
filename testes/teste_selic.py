from dados import CSV, Indicador
import etl
from numpy import random
from sklearn.model_selection import train_test_split

class TesteSelic:
    @property
    def name(self):
        return self.__class__.__name__

    def pega_csv(self):
        csv = CSV.selic
        ipca = Indicador.ipca

        csv["ipcaMes"] = ipca["taxaMes"]
        csv["ipca12Meses"] = ipca["taxa12Meses"]
        csv["ipcaIndice"] = ipca["indice"]
        csv.fillna(method='ffill', inplace=True)
        csv.fillna(method='bfill', inplace=True)
        return csv

    def clusterizacao(self, csv):
        from clusterizacao import run
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from regressaolinear import run
        X = csv[["data", "ipcaMes", "ipca12Meses", "ipcaIndice"]].values
        y = csv["puCompra"].values
        XAmostra = csv[["data", "ipcaMes", "ipca12Meses", "ipcaIndice"]].values[:200, :]
        yAmostra = csv["puCompra"].values[:200]
        return run(csv, X, y, XAmostra, yAmostra)

    def nn_regressao(self, csv):
        from nnregression import run
        X = csv[["data", "ipcaMes", "ipca12Meses", "ipcaIndice"]].values
        y = csv["puCompra"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return run(
            X, y, X_test, y_test, X_train, y_train, hidden_layer_sizes=(1000,500,265,128,),
            activation="relu", solver="lbfgs", learning_rate="adaptive",
            random_state=0
        )

