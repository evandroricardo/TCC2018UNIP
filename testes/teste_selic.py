from data_layer.data import CSV, Indicador
from data_layer import etl
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
        from neural.clusterizacao import run
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from neural.regressaolinear import run
        X = csv[["data", "selic"]].values
        y = csv["puCompra"].values
        XAmostra = csv[["data", "selic"]].values[:200, :]
        yAmostra = csv["puCompra"].values[:200]
        return run(csv, X, y, XAmostra, yAmostra)

    def nn_regressao(self, csv, X_test=None, dont_plot=False):
        from neural.nnregression import run
        X = csv[["data", "selic"]].values
        y = csv["puCompra"].values
        X_train, _X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        if X_test is None:
            X_test = _X_test
        return run(
            X, y, X_test, y_test, X_train, y_train, 
            dont_plot=dont_plot,
            before_plot=self.before_plot_nn_regressao, plot_title="Dias corridos x Preco de compra",
            hidden_layer_sizes=(2000,700,800,1000,),
            activation="relu", solver="adam", learning_rate="adaptive", alpha=0.001, batch_size='auto', 
            learning_rate_init=0.01, power_t=0.01, max_iter=100000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=False,
            early_stopping=False, validation_fraction=0.9999, beta_1=0.505, beta_2=0.999, epsilon=1e-08
        )
    
    def before_plot_nn_regressao(self, nn, yPred, yPredTeste, score_amostra, score_todo):
        preco_selic_2023 = nn.predict([[1674, 0.15]])[0] + 8990
        print("Previsao preco SELIC 2023 no vencimento: {0}; Min: {1}; Max: {2}".format(
            preco_selic_2023,
            preco_selic_2023 * score_todo,
            preco_selic_2023 * (2 - score_todo)
        ))
        return [p + 1772.37 for p in yPred], yPredTeste

    def predict(self, data, selic):
        result = self.nn_regressao(self.pega_csv(), X_test=[[data, selic]], dont_plot=True)
        return result[-1] + 8990, "Rede Neural"
