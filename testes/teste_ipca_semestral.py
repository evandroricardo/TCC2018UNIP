from dados import CSV
from numpy import random
from sklearn.model_selection import train_test_split


class TesteIPCASemestral:
    @property
    def name(self):
        return self.__class__.__name__

    def pega_csv(self):
        return CSV.ipca_jur_semestral

    def clusterizacao(self, csv):
        from clusterizacao import run
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from regressaolinear import run
        X = csv[["data", "taxaCompra"]].values
        y = csv["puCompra"].values
        XAmostra = csv[["data", "taxaCompra"]].values[:200, :]
        yAmostra = csv["puCompra"].values[:200]
        return run(csv, X, y, XAmostra, yAmostra)

    def nn_regressao(self, csv):
        from nnregression import run
        X = csv[["data", "taxaCompra"]].values
        y = csv["puCompra"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return run(
            X, y, X_test, y_test, X_train, y_train, 
            before_plot=self.before_plot_nn_regressao, plot_title="Dias corridos x Preco de compra",
            hidden_layer_sizes=(2000,),  activation='relu', solver='adam', learning_rate='constant', alpha=0.001, 
            batch_size='auto', learning_rate_init=0.01, power_t=0.5, max_iter=10000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=False,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )

    def before_plot_nn_regressao(self, nn, yPred, yPredTeste, score_amostra, score_todo):
        preco_ipcasemestral_2015 = nn.predict([[2937, 0.10]])[0] + 3286.48
        print("Previsao preco IPCA+ Jur Semestral 2026 no vencimento: {0}; Min: {1}; Max: {2}".format(
            preco_ipcasemestral_2015,
            preco_ipcasemestral_2015 * score_todo,
            preco_ipcasemestral_2015 * (2 - score_todo)
        ))
        return yPred, yPredTeste
