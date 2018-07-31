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
        return csv

    def clusterizacao(self, csv):
        from clusterizacao import run
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from regressaolinear import run
        X = csv[["data", "selic"]].values
        y = csv["puCompra"].values
        XAmostra = csv[["data", "selic"]].values[:200, :]
        yAmostra = csv["puCompra"].values[:200]
        return run(csv, X, y, XAmostra, yAmostra)

    def nn_regressao(self, csv):
        from nnregression import run
        X = csv[["data", "selic"]].values
        y = csv["puCompra"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return run(
            X, y, X_test, y_test, X_train, y_train, hidden_layer_sizes=(2000,700,800,1000,),
            activation="relu", solver="adam", learning_rate="adaptive", alpha=0.001, batch_size='auto', 
            learning_rate_init=0.01, power_t=0.01, max_iter=100000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=False,
            early_stopping=False, validation_fraction=0.9999, beta_1=0.505, beta_2=0.999, epsilon=1e-08
        )

