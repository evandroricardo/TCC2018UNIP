from dados import CSV, Indicador
import etl
from numpy import random
from sklearn.model_selection import train_test_split

class TesteSelic:
    @property
    def name(self):
        return self.__class__.__name__

    def pega_csv(self):
        return CSV.selic

    def clusterizacao(self, csv):
        from clusterizacao import run
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from regressaolinear import run
        x = csv["data"].values.reshape(-1, 1)
        y = csv["puCompra"].values.reshape(-1, 1)
        xAmostra = csv["data"][:200].values.reshape(-1, 1)
        yAmostra = csv["puCompra"][:200].values.reshape(-1, 1)
        return run(csv, x, y, xAmostra, yAmostra)

    def nn_regressao(self, csv):
        from nnregression import run
        ipca = Indicador.ipca

        csv["ipcaMes"] = ipca["taxaMes"]
        csv["ipca12Meses"] = ipca["taxa12Meses"]
        csv["ipcaIndice"] = ipca["indice"]
        csv.fillna(method='ffill', inplace=True)
        csv.fillna(method='bfill', inplace=True)

        X = csv[["data", "ipcaMes", "ipca12Meses", "ipcaIndice"]].values
        y = csv["puCompra"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return run(
            X, y, X_test, y_test, X_train, y_train, hidden_layer_sizes=(1000,500,265,128,),
            activation="relu", solver="adam", learning_rate="adaptive", max_iter=1000000000,
            alpha=0.00000000001,
            random_state=0
        )

