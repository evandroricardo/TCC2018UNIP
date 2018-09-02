from data_layer.data import CSV
from numpy import random
from sklearn.model_selection import train_test_split


class TesteIPCA:
    @property
    def name(self):
        return self.__class__.__name__

    def pega_csv(self):
        return CSV.ipca

    def clusterizacao(self, csv):
        from neural.clusterizacao import run
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from neural.regressaolinear import run
        X = csv[["data", "taxaCompra"]].values
        y = csv["puCompra"].values
        XAmostra = csv[["data", "taxaCompra"]].values[:200, :]
        yAmostra = csv["puCompra"].values[:200]
        return run(csv, X, y, XAmostra, yAmostra)

    def nn_regressao(self, csv, X_test=None, dont_plot=False):
        from neural.nnregression import run
        X = csv[["data", "taxaCompra"]].values
        y = csv["puCompra"].values
        X_train, _X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        if X_test is None:
            X_test = _X_test
        return run(
            X, y, X_test, y_test, X_train, y_train, 
            dont_plot=dont_plot,
            before_plot=self.before_plot_nn_regressao, plot_title="Dias corridos x Preco de compra",
            hidden_layer_sizes=(100,),  activation='relu', solver='adam', learning_rate='constant', alpha=0.001, 
            batch_size='auto', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=False,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )

    def before_plot_nn_regressao(self, nn, yPred, yPredTeste, score_amostra, score_todo):
        preco_ipca_2024 = nn.predict([[2207, 0.0494]])[0] + 2119.26
        print("Previsao preco IPCA+ 2024 no vencimento: {0}; Min: {1}; Max: {2}".format(
            preco_ipca_2024,
            preco_ipca_2024 * score_todo,
            preco_ipca_2024 * (2 - score_todo)
        ))
        return yPred, yPredTeste

    def predict(self, data, taxa_compra, K=0):
        result = self.nn_regressao(self.pega_csv(), X_test=[[data, taxa_compra]], dont_plot=True)
        return result[-1] + K, "Rede Neural"
